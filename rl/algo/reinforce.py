import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import time

class REINFORCE:
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True,
                 use_recurrent_generator=False,
                 decoder_rl_loss_coef = 1.0,
                 condition_rl_loss_coef = 0.0,
                 latent_rl_loss_coef = 0.0,
                 setup='RL',
                 use_mean_only=False,
                 writer=None):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.entropy_coef = entropy_coef
        self.decoder_rl_loss_coef = decoder_rl_loss_coef
        self.condition_rl_loss_coef = condition_rl_loss_coef
        self.latent_rl_loss_coef = latent_rl_loss_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.use_recurrent_generator = use_recurrent_generator

        self.setup = setup
        self.use_mean_only = use_mean_only
        if setup == 'RL':
            self.optimizer = optim.Adam(list(actor_critic.meta_controller.parameters()) +
                                         list(actor_critic.program_vae.vae._enc_mu.parameters()) +
                                         #list(actor_critic.program_vae.vae.decoder.parameters()) +
                                         list(actor_critic.program_vae.vae._enc_log_sigma.parameters()),
                                        lr=lr, eps=eps)
        elif setup == 'supervisedRL':
            self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
        else:
            raise NotImplementedError()
        assert setup == 'supervisedRL' or setup == 'RL'

        self._global_step = 0
        self.writer = writer

    def update(self, rollouts, use_decoder_dist=True):
        returns = rollouts.returns[:-1]
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        self.writer.add_scalar('ppo/returns', returns.mean(), self._global_step)

        if self.condition_rl_loss_coef > 0.0:
            agent_action_returns = rollouts.agent_action_returns[:-1]
            agent_action_returns = (agent_action_returns - agent_action_returns.mean()) / (agent_action_returns.std() + 1e-5)
            self.writer.add_scalar('ppo/agent_action_returns', agent_action_returns.mean(), self._global_step)

        value_loss_epoch = 0
        action_loss_epoch = 0
        condition_loss_epoch = 0
        latent_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            data_generator = rollouts.feed_forward_generator(None, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, value_preds_batch, return_batch, masks_batch, \
                old_action_log_probs_batch, adv_targ, output_masks_batch, eop_actions_batch, agent_actions_batch,\
                agent_value_preds_batch, agent_action_return_batch, agent_action_masks_batch,\
                old_agent_action_log_probs_batch, agent_action_adv_targ, program_embeddings_batch,\
                agent_initial_states_batch, old_latent_log_probs_batch, program_ids_batch = sample
                return_batch = return_batch.view(-1, 1)

                # Reshape to do in a single forward pass for all steps
                obs_batch = (obs_batch, agent_initial_states_batch, program_embeddings_batch)
                _, action_log_probs, dist_entropy, _, _, _, agent_action_log_probs, agent_action_dist_entropy,\
                distribution_params, latent_log_probs,\
                latent_dist_entropy = self.actor_critic.evaluate_actions(obs_batch, recurrent_hidden_states_batch,
                                                                         masks_batch, actions_batch, output_masks_batch,
                                                                         eop_actions_batch, agent_actions_batch,
                                                                         agent_action_masks_batch, program_ids_batch,
                                                                         deterministic=not use_decoder_dist)

                # calculate action loss G_t * log_probs
                assert return_batch.shape == action_log_probs.shape
                if use_decoder_dist:
                    action_loss = -(return_batch.detach() * action_log_probs).mean()
                    dist_entropy = dist_entropy
                else:
                    action_loss = -(return_batch.detach() * latent_log_probs).mean()
                    dist_entropy = latent_dist_entropy

                if self.condition_rl_loss_coef > 0.0:
                    agent_action_loss = -(agent_action_return_batch.view(-1,1).detach() * agent_action_log_probs.view(-1, 1)).mean()

                self.optimizer.zero_grad()
                final_loss = torch.tensor(0.0, requires_grad=True, device=action_loss.device)
                if self.decoder_rl_loss_coef > 0.0 and self.condition_rl_loss_coef > 0.0:
                    loss = action_loss - dist_entropy * self.entropy_coef
                    agent_action_loss = agent_action_loss - agent_action_dist_entropy * self.entropy_coef
                    final_loss = self.decoder_rl_loss_coef * loss + self.condition_rl_loss_coef * agent_action_loss
                elif self.decoder_rl_loss_coef > 0.0:
                    loss = action_loss - dist_entropy * self.entropy_coef
                    final_loss = loss
                elif self.decoder_rl_loss_coef > 0.0:
                    agent_action_loss = agent_action_loss - agent_action_dist_entropy * self.entropy_coef
                    final_loss = agent_action_loss
                else:
                    print('Warning! using only latent loss for training')
                    # assert 0, 'Invalid RL loss coefficients'

                if self.latent_rl_loss_coef > 0.0:
                    net = self.actor_critic.program_vae.vae if self.setup == 'RL' else self.actor_critic.vae
                    if self.use_mean_only:
                        latent_loss = 0.5 * torch.mean(net.z_mean * net.z_mean)
                    else:
                        latent_loss = net.latent_loss(net.z_mean, net.z_sigma)
                    final_loss += self.latent_rl_loss_coef * latent_loss

                final_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                action_loss_epoch += action_loss.detach().cpu().numpy()
                condition_loss_epoch += agent_action_loss.detach().cpu().numpy() if self.condition_rl_loss_coef > 0.0 else 0
                latent_loss_epoch += latent_loss.detach().cpu().numpy() if self.latent_rl_loss_coef > 0.0 else 0
                dist_entropy_epoch += dist_entropy.detach().cpu().numpy()
                del loss

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        self.writer.add_scalar('ppo/avg_decoder_loss', action_loss_epoch, self._global_step)
        self.writer.add_scalar('ppo/avg_condition_loss', condition_loss_epoch, self._global_step)
        self.writer.add_scalar('ppo/avg_latent_loss', latent_loss_epoch, self._global_step)
        self.writer.add_scalar('ppo/entropy', dist_entropy_epoch, self._global_step)
        self.writer.add_scalar('ppo/entropy_x_entropy_coef', dist_entropy_epoch * self.entropy_coef, self._global_step)
        self._global_step += 1

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch

