import time
import numpy as np

import torch
import torch.nn as nn

from pretrain.misc_utils import log_record_dict, create_directory
from pretrain.SupervisedModel import SupervisedModel
from pretrain.RLModel import RLModel2


class SupervisedRLModel(object):
    def __init__(self, *args, **kwargs):
        super(SupervisedRLModel, self).__init__()
        self.sl_model = SupervisedModel(*args, **kwargs)
        self.rl_model = RLModel2(self.sl_model.net, *args, **kwargs)
        # self.rl_model.net.program_vae = self.sl_model.net
        self.num_program_tokens = self.rl_model.net.num_program_tokens
        self.config = self.sl_model.config
        self.writer = self.rl_model.writer
        self.do_supervised = self.config['do_supervised']
        self.do_rl = self.config['do_RL']

        self.train_rl_update_num = 0
        self.eval_rl_update_num = 0
        self.rl_update_ratio = self.config['rl']['algo']['ppo']['ppo_epoch']/2 if self.config['rl']['algo']['name'] == 'ppo' else 1
        self.sl_update_count = 0

    def train(self,  train_dataloader, val_dataloader, r_train_dataloader, r_eval_dataloader, *args, **kwargs):
        tr_loader = train_dataloader
        val_loader = val_dataloader

        # Initialize params
        max_epoch = kwargs['max_epoch']

        # Train epochs
        best_valid_loss = np.inf
        best_valid_epoch = 0

        # RL variables
        start = time.time()
        config = self.config
        max_update_num = self.rl_model.num_updates

        # update counts
        supervised_updates_in_one_epoch = len(train_dataloader)
        rl_updates_in_one_update_step = self.rl_model.agent.num_mini_batch
        rl_updates_in_one_epoch = supervised_updates_in_one_epoch//rl_updates_in_one_update_step

        for epoch in range(max_epoch):


            if self.do_supervised:
                self.sl_model.net.condition_policy.setup = 'supervised'
                self.sl_model.net.vae.decoder.setup = 'supervised'
                best_valid_epoch, best_valid_loss, \
                record_dict_eval, done = self.sl_model.run_one_epoch(epoch, best_valid_epoch, best_valid_loss, tr_loader,
                                                                    val_loader, *args, **kwargs)
                self.sl_model.net.condition_policy.setup = 'supervisedRL'
                self.sl_model.net.vae.decoder.setup = 'supervisedRL'
                self.sl_model.net.train()
                torch.set_grad_enabled(True)

                if record_dict_eval['mean_decoder_greedy_program_accuracy'] < 80:
                    continue
            else:
                self.sl_model.net.condition_policy.setup = 'supervised'
                self.sl_model.net.vae.decoder.setup = 'supervised'
                self.sl_model.net.eval()
                torch.set_grad_enabled(False)
                self.sl_model.evaluate(val_loader, epoch, *args, **kwargs)
                self.sl_model.net.condition_policy.setup = 'supervisedRL'
                self.sl_model.net.vae.decoder.setup = 'supervisedRL'
                self.sl_model.net.train()
                torch.set_grad_enabled(True)



            if self.do_rl and self.sl_update_count % self.rl_update_ratio == 0:
                epoch_train_rewards = []
                for batch_idx, batch in enumerate(r_train_dataloader):
                    programs, ids, programs_mask, s_h, a_h, a_h_len = batch
                    programs = programs.view(self.rl_model.num_steps, self.rl_model.num_processes, *programs.shape[1:])
                    programs_mask = programs_mask.view(self.rl_model.num_steps, self.rl_model.num_processes, *programs_mask.shape[1:])
                    initial_states = s_h.view(self.rl_model.num_steps, self.rl_model.num_processes, *s_h.shape[1:])
                    a_h = a_h.view(self.rl_model.num_steps, self.rl_model.num_processes, *a_h.shape[1:])
                    a_h_len = a_h_len.view(self.rl_model.num_steps, self.rl_model.num_processes, *a_h_len.shape[1:])
                    self.rl_model.rollouts.obs[:self.rl_model.num_steps] = programs[:,:,:-1]
                    self.rl_model.rollouts.obs[-1] = programs[-1,:,:-1]

                    assert self.rl_model.rollouts.agent_initial_states.shape == initial_states.shape
                    self.rl_model.rollouts.agent_initial_states = initial_states
                    self.rl_model.rollouts.gt_agent_actions = a_h

                    self.rl_model.net.condition_policy.setup = 'RL'
                    self.rl_model.net.vae.decoder.setup = 'RL'
                    self.rl_model.update_step(self.train_rl_update_num, start, config)
                    self.train_rl_update_num += 1
                    self.sl_model.net.condition_policy.setup = 'supervisedRL'
                    self.sl_model.net.vae.decoder.setup = 'supervisedRL'

                    epoch_train_rewards.append(self.rl_model.rollouts.rewards.mean().detach().cpu().numpy())

                print("current_update_num: ", self.train_rl_update_num)
                self.writer.add_scalar('agent/epoch_train_reward_mean', np.array(epoch_train_rewards).mean(), epoch)
            self.sl_update_count += 1

            if r_eval_dataloader is not None:
                self.rl_evaluate(r_eval_dataloader, epoch)

        return

    def rl_evaluate(self, r_eval_dataloader, epoch):
        eval_rl_rewards = torch.zeros(self.rl_model.num_steps, self.rl_model.num_processes, 1)
        eval_rl_distribution_params = torch.zeros(self.rl_model.num_steps, self.rl_model.num_processes, 2,
                                                  self.rl_model.net.recurrent_hidden_state_size)

        self.rl_model.decoder_envs.reset()
        epoch_eval_rewards = []
        for batch_idx, batch in enumerate(r_eval_dataloader):
            t = time.time()
            programs, ids, programs_mask, s_h, a_h, a_h_len = batch
            programs = programs.view(self.rl_model.num_steps, self.rl_model.num_processes, *programs.shape[1:])
            initial_states = s_h.view(self.rl_model.num_steps, self.rl_model.num_processes, *s_h.shape[1:])
            self.rl_model.net.condition_policy.setup = 'RL'
            self.rl_model.net.vae.decoder.setup = 'RL'

            # Sample actions
            for step in range(self.rl_model.num_steps):
                with torch.no_grad():
                    outputs = self.rl_model.actor_critic.act(programs[step, :, :-1],
                                                             self.rl_model.rollouts.recurrent_hidden_states[step],
                                                             self.rl_model.rollouts.masks[step],
                                                             initial_states[step],
                                                             deterministic=not self.config['net']['controller'][
                                                                 'use_decoder_dist'])
                    value, pred_programs, _, recurrent_hidden_states, _, _, agent_value, _, _, _, \
                    program_embeddings, distribution_params, _, _, _, _ = outputs

                # Observation reward and next obs
                _, reward, done, infos = self.rl_model._supervisedRL_env_step(pred_programs,
                                                                              programs[step, :, :-1],
                                                                              initial_states[step])

                eval_rl_rewards[step] = reward
                eval_rl_distribution_params[step] = distribution_params

            self.sl_model.net.condition_policy.setup = 'supervisedRL'
            self.sl_model.net.vae.decoder.setup = 'supervisedRL'

            # logging
            cum_reward = eval_rl_rewards.cpu().numpy()
            self.writer.add_scalar('agent/eval_reward_sum', cum_reward.sum(), self.eval_rl_update_num)
            self.writer.add_scalar('agent/eval_reward_max', cum_reward.max(), self.eval_rl_update_num)
            self.writer.add_scalar('agent/eval_reward_min', cum_reward.min(), self.eval_rl_update_num)
            self.writer.add_scalar('agent/eval_reward_mean', cum_reward.mean(), self.eval_rl_update_num)
            self.writer.add_scalar('eval_distribution/mean',
                                   torch.norm(eval_rl_distribution_params[:, 0, :].detach(), dim=-1,
                                              p=2).mean().cpu().numpy(), self.eval_rl_update_num)
            self.writer.add_scalar('eval_distribution/std',
                                   torch.norm(eval_rl_distribution_params[:, 1, :].detach(), dim=-1,
                                              p=2).mean().cpu().numpy(), self.eval_rl_update_num)
            # log all items in dict
            record_dict = {'udpate_num': self.eval_rl_update_num,
                           'update_reward_mean': cum_reward.mean(),
                           'update_reward_sum': cum_reward.sum(), 'update_reward_min': cum_reward.min(),
                           'update_reward_max': cum_reward.max()}
            log_record_dict('eval', record_dict, self.rl_model.global_logs)
            if self.rl_model.verbose:
                self.rl_model._print_record_dict(record_dict, self.eval_rl_update_num, self.rl_model.num_updates, 'eval',
                                                 time.time() - t)
            self.eval_rl_update_num += 1

            epoch_eval_rewards.append(cum_reward.mean())

        self.writer.add_scalar('agent/epoch_eval_reward_mean', np.array(epoch_eval_rewards).mean(), epoch)
        return



