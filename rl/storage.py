# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/storage.py
import time
import collections
from itertools import chain

import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage2(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space,
                 recurrent_hidden_state_size, execution_guided, num_demo_per_program, max_demo_length,
                 exec_env_state_shape, dense_execution_reward=False, future_rewards=True):
        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.recurrent_hidden_states = torch.zeros(
            num_steps + 1, num_processes, recurrent_hidden_state_size)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.rewards_intrinsic = torch.zeros(num_steps, num_processes, 1)
        self.rewards_env = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        self.eop_actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)
        # NOTE: since we are toggling output_masks a lot, make sure they are torch.bool
        self.output_masks = torch.ones(num_steps, num_processes, action_shape).to(torch.bool)

        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)

        self.num_steps = num_steps
        self.num_processes = num_processes
        self.future_rewards = future_rewards
        self.num_demo_per_program = num_demo_per_program
        self.step = 0

        # for intention space
        self.program_embeddings = torch.zeros(num_steps, num_processes, recurrent_hidden_state_size)
        self.program_distribution_params = torch.zeros(num_steps, num_processes, 2, recurrent_hidden_state_size)
        h, w, c = exec_env_state_shape
        self.agent_initial_states = torch.zeros((num_steps, num_processes, num_demo_per_program, 1, c, h, w))
        self.agent_value_preds = torch.zeros(num_steps + 1, num_processes, num_demo_per_program, 1)
        self.agent_actions = torch.zeros(num_steps, num_processes, num_demo_per_program, max_demo_length - 1)
        self.agent_action_rewards = torch.zeros(num_steps, num_processes, num_demo_per_program, 1)
        self.agent_action_returns = torch.zeros(num_steps + 1, num_processes, num_demo_per_program, 1)
        self.agent_action_log_probs = torch.zeros(num_steps, num_processes, num_demo_per_program, 1)
        self.agent_action_masks = torch.ones(num_steps, num_processes, num_demo_per_program, max_demo_length - 1).to(torch.bool)
        self.original_masks = torch.ones(num_steps + 1, num_processes, 1)
        self.latent_log_probs = torch.zeros(num_steps, num_processes, 1)

        # debug ids
        self._debug_ids = [None for i in range(num_steps)]

    def to(self, device):
        self.obs = self.obs.to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.rewards_intrinsic = self.rewards_intrinsic.to(device)
        self.rewards_env = self.rewards_env.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.eop_actions = self.eop_actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)
        self.output_masks = self.output_masks.to(device)
        self.program_embeddings = self.program_embeddings.to(device)
        self.program_distribution_params = self.program_distribution_params.to(device)
        self.agent_initial_states = self.agent_initial_states.to(device)
        self.agent_value_preds = self.agent_value_preds.to(device)
        self.agent_actions = self.agent_actions.to(device)
        self.agent_action_rewards = self.agent_action_rewards.to(device)
        self.agent_action_returns = self.agent_action_returns.to(device)
        self.agent_action_log_probs = self.agent_action_log_probs.to(device)
        self.agent_action_masks = self.agent_action_masks.to(device)
        self.original_masks = self.original_masks.to(device)
        self.latent_log_probs = self.latent_log_probs.to(device)

    def insert(self, obs, recurrent_hidden_states, actions, action_log_probs, value_preds, rewards, rewards_intrinsic,
               rewards_env, masks, bad_masks, output_masks, eop_actions, agent_value_preds,agent_actions,
               agent_action_log_probs, agent_action_rewards, agent_action_masks, program_embeddings,
               agent_initial_states, distribution_params, original_masks, latent_log_probs, debug_dict=None):
        self.obs[self.step + 1].copy_(obs)
        self.recurrent_hidden_states[self.step + 1].copy_(recurrent_hidden_states)
        self.actions[self.step].copy_(actions)
        self.eop_actions[self.step].copy_(eop_actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.rewards_env[self.step].copy_(rewards_env)
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)
        self.output_masks[self.step].copy_(output_masks)

        self.agent_value_preds[self.step].copy_(agent_value_preds)
        self.agent_actions[self.step].copy_(agent_actions)
        self.agent_action_log_probs[self.step].copy_(agent_action_log_probs)
        self.agent_action_rewards[self.step].copy_(agent_action_rewards)
        self.agent_action_masks[self.step].copy_(agent_action_masks)
        self.program_embeddings[self.step].copy_(program_embeddings)
        self.agent_initial_states[self.step].copy_(agent_initial_states)
        self.program_distribution_params[self.step].copy_(distribution_params)
        self.original_masks[self.step + 1].copy_(original_masks)
        self.latent_log_probs[self.step].copy_(latent_log_probs)

        if rewards_intrinsic is not None:
            self.rewards_intrinsic[self.step].copy_(rewards_intrinsic)
        if debug_dict is not None:
            self._debug_ids[self.step] = debug_dict['ids']

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])

        self.original_masks[0].copy_(self.original_masks[-1])

    def compute_returns(self, next_value,
                        agent_action_next_value,
                        use_gae,
                        gamma,
                        gae_lambda,
                        use_proper_time_limits=True,
                        algorithm='ppo'):
        if use_proper_time_limits:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = self.rewards[step] + gamma * self.value_preds[
                        step + 1] * self.masks[step +
                                               1] - self.value_preds[step]
                    gae = delta + gamma * gae_lambda * self.masks[step +
                                                                  1] * gae
                    gae = gae * self.bad_masks[step + 1]
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = (self.returns[step + 1] * \
                                          gamma * self.masks[step + 1] + self.rewards[step]) * self.bad_masks[step + 1] \
                                         + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
        else:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = self.rewards[step] + gamma * self.value_preds[
                        step + 1] * self.masks[step +
                                               1] - self.value_preds[step]
                    gae = delta + gamma * gae_lambda * self.masks[step +
                                                                  1] * gae
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = self.returns[step + 1] * \
                                         gamma * self.masks[step + 1] + self.rewards[step]

        # TODO: properly setup masks for returns for condition policy if you want to propagate gradients through it
        if use_proper_time_limits:
            if use_gae:
                self.agent_value_preds[-1] = agent_action_next_value
                gae = 0
                for step in reversed(range(self.agent_action_rewards.size(0))):
                    delta = self.agent_action_rewards[step] + gamma * self.agent_value_preds[
                        step + 1] * self.masks[step +
                                               1] - self.agent_value_preds[step]
                    gae = delta + gamma * gae_lambda * self.masks[step +
                                                                  1] * gae
                    gae = gae * self.bad_masks[step + 1]
                    self.agent_action_returns[step] = gae + self.agent_value_preds[step]
            else:
                self.agent_action_returns[-1] = agent_action_next_value
                for step in reversed(range(self.agent_action_rewards.size(0))):
                    self.agent_action_returns[step] = (self.agent_action_returns[step + 1] * \
                                          gamma * self.masks[step + 1] + self.agent_action_rewards[step]) * self.bad_masks[step + 1] \
                                         + (1 - self.bad_masks[step + 1]) * self.agent_value_preds[step]
        else:
            if use_gae:
                self.agent_value_preds[-1] = agent_action_next_value
                gae = 0
                for step in reversed(range(self.agent_action_rewards.size(0))):
                    delta = self.agent_action_rewards[step] + gamma * self.agent_value_preds[
                        step + 1] * self.masks[step +
                                               1].unsqueeze(1).repeat(1, self.num_demo_per_program,1) - self.agent_value_preds[step]
                    gae = delta + gamma * gae_lambda * self.masks[step +
                                                                  1].unsqueeze(1).repeat(1, self.num_demo_per_program,1) * gae
                    self.agent_action_returns[step] = gae + self.agent_value_preds[step]
            else:
                self.agent_action_returns[-1] = agent_action_next_value
                for step in reversed(range(self.agent_action_rewards.size(0))):
                    #self.agent_action_returns[step] = self.agent_action_returns[step + 1] * \
                    #                     gamma * self.masks[step + 1] + self.agent_action_rewards[step]
                    self.agent_action_returns[step] = self.agent_action_returns[step + 1] * \
                                                      gamma + self.agent_action_rewards[step]

    def feed_forward_generator(self,
                               advantages,
                               num_mini_batch=None,
                               mini_batch_size=None,
                               agent_action_advantages=None):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_processes, num_steps, num_processes * num_steps,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)
        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(
                -1, self.recurrent_hidden_states.size(-1))[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            eop_actions_batch = self.eop_actions.view(-1, self.eop_actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            output_masks_batch = self.output_masks.view(-1, self.output_masks.size(-1))[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]

            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

            # num_steps, num_processes, num_demo_per_program, max_demo_length - 1
            x = self.agent_actions.shape
            if agent_action_advantages is None:
                agent_action_adv_targ = None
            else:
                agent_action_adv_targ = agent_action_advantages.view(-1, x[-2], 1)[indices]
            agent_value_preds_batch = self.agent_value_preds[:-1].view(-1, x[-2], 1)[indices]
            agent_actions_batch = (self.agent_actions.view(-1, x[-2], x[-1])[indices]).view(-1, x[-1])
            agent_action_return_batch = self.agent_action_returns[:-1].view(-1, x[-2], 1)[indices]
            agent_action_masks_batch = (self.agent_action_masks.view(-1, x[-2], x[-1])[indices]).view(-1, x[-1])
            old_agent_action_log_probs_batch = self.agent_action_log_probs.view(-1, x[-2], 1)[indices]
            program_embeddings_batch = self.program_embeddings.view(-1, self.program_embeddings.shape[-1])[indices]
            x = self.agent_initial_states.shape
            agent_initial_states_batch = self.agent_initial_states.view(x[0]*x[1], *x[2:])[indices]

            old_latent_log_probs_batch = self.latent_log_probs.view(-1, 1)[indices]

            # debug info
            program_ids_batch = None
            if self._debug_ids[0] is not None:
                program_ids_batch = [list(chain.from_iterable(self._debug_ids))[idx] for idx in indices]

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, value_preds_batch, return_batch, masks_batch,\
                  old_action_log_probs_batch, adv_targ, output_masks_batch, eop_actions_batch, agent_actions_batch,\
                  agent_value_preds_batch, agent_action_return_batch, agent_action_masks_batch, old_agent_action_log_probs_batch,\
                  agent_action_adv_targ, program_embeddings_batch, agent_initial_states_batch,\
                  old_latent_log_probs_batch, program_ids_batch
