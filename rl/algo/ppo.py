import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class PPO:
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
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

        self.value_loss_coef = value_loss_coef
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
                                         #list(actor_critic.program_vae.vae.decoder.critic.parameters()) +
                                         #list(actor_critic.program_vae.vae.decoder.critic_linear.parameters()) +
                                         list(actor_critic.program_vae.vae._enc_log_sigma.parameters()),
                                        lr=lr, eps=eps)
        elif setup == 'supervisedRL':
            self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
        else:
            raise NotImplementedError()
        assert setup == 'supervisedRL' or setup == 'RL'

        #self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

        self._global_step = 0
        self.writer = writer

    def _calculate_loss(self, action_log_probs, old_action_log_probs_batch, adv_targ, value_preds_batch,
                                     values, return_batch):
        action_log_probs = action_log_probs.view(-1, 1)
        old_action_log_probs_batch = old_action_log_probs_batch.view(-1, 1)
        adv_targ = adv_targ.view(-1, 1)
        value_preds_batch = value_preds_batch.view(-1, 1)
        values = values.view(-1, 1)
        return_batch = return_batch.view(-1, 1)

        ratio = torch.exp(action_log_probs -
                          old_action_log_probs_batch)
        surr1 = ratio * adv_targ
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                            1.0 + self.clip_param) * adv_targ
        action_loss = -torch.min(surr1, surr2).mean()

        if self.use_clipped_value_loss:
            value_pred_clipped = value_preds_batch + \
                                 (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
            value_losses = (values - return_batch).pow(2)
            value_losses_clipped = (
                    value_pred_clipped - return_batch).pow(2)
            value_loss = 0.5 * torch.max(value_losses,
                                         value_losses_clipped).mean()
        else:
            value_loss = 0.5 * (return_batch - values).pow(2).mean()
        return action_loss, value_loss

    def update(self, rollouts, use_decoder_dist=True):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)
        self.writer.add_scalar('ppo/advantages', advantages.mean(), self._global_step)

        agent_action_advantages = None
        if self.condition_rl_loss_coef > 0.0:
            agent_action_advantages = rollouts.agent_action_returns[:-1] - rollouts.agent_value_preds[:-1]
            agent_action_advantages = (agent_action_advantages - agent_action_advantages.mean()) / (
                    agent_action_advantages.std() + 1e-5)
            self.writer.add_scalar('ppo/agent_action_advantages', agent_action_advantages.mean(), self._global_step)

        value_loss_epoch = 0
        action_loss_epoch = 0
        condition_loss_epoch = 0
        latent_loss_epoch = 0
        condition_loss_epoch = 0
        latent_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent and self.use_recurrent_generator:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
                assert 0, "policy doesn't support this yet"
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch, agent_action_advantages=agent_action_advantages)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, value_preds_batch, return_batch, masks_batch, \
                old_action_log_probs_batch, adv_targ, output_masks_batch, eop_actions_batch, agent_actions_batch, \
                agent_value_preds_batch, agent_action_return_batch, agent_action_masks_batch, \
                old_agent_action_log_probs_batch, agent_action_adv_targ, program_embeddings_batch,\
                agent_initial_states_batch, old_latent_log_probs_batch, program_ids_batch = sample

                # Reshape to do in a single forward pass for all steps
                obs_batch = (obs_batch, agent_initial_states_batch, program_embeddings_batch)
                evaluate_outputs = self.actor_critic.evaluate_actions(obs_batch, recurrent_hidden_states_batch,
                                                                         masks_batch, actions_batch, output_masks_batch,
                                                                         eop_actions_batch, agent_actions_batch,
                                                                         agent_action_masks_batch, program_ids_batch,
                                                                         deterministic=not use_decoder_dist)
                values, action_log_probs, dist_entropy, _, _, agent_values, agent_action_log_probs, \
                agent_action_dist_entropy, _, latent_log_probs, latent_dist_entropy = evaluate_outputs

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                if self.condition_rl_loss_coef > 0.0:
                    agent_action_loss, agent_value_loss = self._calculate_loss(agent_action_log_probs,
                                                                               old_agent_action_log_probs_batch,
                                                                               agent_action_adv_targ,
                                                                               agent_value_preds_batch,
                                                                               agent_values, agent_action_return_batch)

                self.optimizer.zero_grad()
                loss = value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef
                if self.condition_rl_loss_coef > 0.0:
                    agent_loss = agent_value_loss * self.value_loss_coef + agent_action_loss - agent_action_dist_entropy * self.entropy_coef
                    loss += agent_loss
                if self.latent_rl_loss_coef > 0.0:
                    net = self.actor_critic.program_vae.vae if self.setup == 'RL' else self.actor_critic.vae
                    if self.use_mean_only:
                        latent_loss = 0.5 * torch.mean(net.z_mean * net.z_mean)
                    else:
                        latent_loss = net.latent_loss(net.z_mean, net.z_sigma)
                    loss += self.latent_rl_loss_coef * latent_loss

                loss = loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.detach().cpu().numpy()
                action_loss_epoch += action_loss.detach().cpu().numpy()
                dist_entropy_epoch += dist_entropy.detach().cpu().numpy()
                condition_loss_epoch += agent_action_loss.detach().cpu().numpy() if self.condition_rl_loss_coef > 0.0 else 0
                latent_loss_epoch += latent_loss.detach().cpu().numpy() if self.latent_rl_loss_coef > 0.0 else 0
                del loss

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        self.writer.add_scalar('ppo/avg_policy_loss', action_loss_epoch, self._global_step)
        self.writer.add_scalar('ppo/avg_value_loss', value_loss_epoch, self._global_step)
        self.writer.add_scalar('ppo/entropy', dist_entropy_epoch, self._global_step)
        self.writer.add_scalar('ppo/value_loss_x_value_coef', value_loss_epoch * self.value_loss_coef, self._global_step)
        self.writer.add_scalar('ppo/entropy_x_entropy_coef', dist_entropy_epoch * self.entropy_coef, self._global_step)
        self.writer.add_scalar('ppo/avg_condition_loss', condition_loss_epoch, self._global_step)
        self.writer.add_scalar('ppo/avg_latent_loss', latent_loss_epoch, self._global_step)
        self._global_step += 1

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch

