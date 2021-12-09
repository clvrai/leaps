import os
import time
import pickle
import shutil
import time
from collections import defaultdict
from collections import deque
from multiprocessing import Pool
import numpy as np

import torch
import torch.nn as nn
from torch.optim import lr_scheduler

from prl_gym.program_env import ProgramEnv1
from pretrain.misc_utils import log_record_dict, create_directory
from rl import utils
from rl.algo.ppo import PPO
from rl.algo.a2c_acktr import A2C_ACKTR
from rl.algo.reinforce import REINFORCE
from rl.envs import make_vec_envs
from rl.storage import RolloutStorage2
from utils.misc_utils import HyperParameterScheduler


class BaseRLModel(object):

    def __init__(self, Net, program_vae_net, device, config, dummy_envs, dsl, logger, writer, global_logs, verbose):
        self.device = device
        self.config = config
        self.global_logs = global_logs
        self.verbose = verbose
        self.logger = logger
        self.writer = writer
        self.envs = dummy_envs
        self.dsl = dsl

        log_dir = os.path.expanduser(os.path.join(config['outdir'], 'openai'))
        eval_log_dir = log_dir + "_eval"
        utils.cleanup_log_dir(log_dir)
        utils.cleanup_log_dir(eval_log_dir)

        cfg_rl = config['rl']
        cfg_algo = cfg_rl['algo']
        cfg_envs = config['rl']['envs']

        custom = True if "karel" or "CartPoleDiscrete" in cfg_envs['executable']['name'] else False
        logger.info('Using environment: {}'.format(cfg_envs['executable']['name']))
        self.envs = make_vec_envs(cfg_envs['executable']['name'], config['seed'], cfg_rl['num_processes'],
                                  cfg_rl['gamma'], os.path.join(config['outdir'], 'openai'), device, False,
                                  custom_env=custom, custom_env_type='program', custom_kwargs={'config': config['args']})

        decoder_log_dir = os.path.expanduser(os.path.join(config['outdir'], 'openai_decoder'))
        decoder_eval_log_dir = decoder_log_dir + "_eval"
        utils.cleanup_log_dir(decoder_log_dir)
        utils.cleanup_log_dir(decoder_eval_log_dir)
        if self.config['algorithm'] == 'supervisedRL':
            self.decoder_envs = make_vec_envs(cfg_envs['executable']['name'], config['seed'], cfg_rl['num_processes'],
                                              cfg_rl['gamma'], os.path.join(config['outdir'], 'openai_decoder'), device,
                                              True, custom_env=custom, custom_env_type='program',
                                              custom_kwargs={'config': config['args'], 'metadata':{'alpha': 2}})
            self.decoder_envs.reset()

        exec_env_states = self.envs.render(mode='init_states')
        exec_env_state_shape = exec_env_states[0].shape if cfg_rl['num_processes'] == 1 else exec_env_states[0][1].shape

        gt_program_str = open(cfg_envs['executable']['task_file']).readlines()[0].strip()
        if cfg_envs['executable']['task_definition'] == 'program':
            logger.info('\n=== Ground-truth program === \n{}\n'
                        '============================'.format(gt_program_str))
            writer.add_text('program/gt', gt_program_str, 0)

        # build policy network
        if Net is not None:
            self.net = self.actor_critic = Net(self.envs, **config)
        else:
            self.net = self.actor_critic = program_vae_net
        self.actor_critic.to(device)
        logger.info('Policy Network:\n{}'.format(self.actor_critic))
        logger.info('Policy Network total parameters: {}'.format(utils.count_parameters(self.actor_critic)))

        # Load parameters if available
        ckpt_path = config['net']['saved_params_path']
        sup_params_path = config['net']['saved_sup_params_path']
        if ckpt_path is not None:
            self.load_net(ckpt_path)
        if sup_params_path is not None:
            assert ckpt_path is None, 'conflict in loading weights from supervised training and previous RL training'
            self.load_net2(sup_params_path)

        if cfg_rl['algo']['name'] == 'a2c':
            cfg_a2c = config['rl']['algo']['a2c']
            self.agent = A2C_ACKTR(
                self.actor_critic,
                cfg_rl['algo']['value_loss_coef'],
                cfg_rl['algo']['entropy_coef'],
                lr=cfg_rl['algo']['lr'],
                eps=cfg_a2c['eps'],
                alpha=cfg_a2c['alpha'],
                max_grad_norm=cfg_algo['max_grad_norm'],
                use_recurrent_generator=cfg_rl['algo']['use_recurrent_generator'],
                writer=writer)
        elif cfg_rl['algo']['name'] == 'ppo':
            cfg_ppo = config['rl']['algo']['ppo']
            self.agent = PPO(
                self.actor_critic,
                cfg_ppo['clip_param'],
                cfg_ppo['ppo_epoch'],
                cfg_ppo['num_mini_batch'],
                cfg_rl['algo']['value_loss_coef'],
                cfg_rl['algo']['entropy_coef'],
                lr=cfg_rl['algo']['lr'],
                eps=cfg_ppo['eps'],
                max_grad_norm=cfg_algo['max_grad_norm'],
                use_recurrent_generator=cfg_rl['algo']['use_recurrent_generator'],
                decoder_rl_loss_coef=cfg_rl['loss']['decoder_rl_loss_coef'],
                condition_rl_loss_coef=cfg_rl['loss']['condition_rl_loss_coef'],
                latent_rl_loss_coef=cfg_rl['loss']['latent_rl_loss_coef'],
                setup=config['algorithm'],
                use_mean_only=cfg_rl['loss']['use_mean_only_for_latent_loss'],
                writer=writer)
        elif cfg_rl['algo']['name'] == 'acktr':
            cfg_acktr = config['rl']['algo']['acktr']
            self.agent = A2C_ACKTR(
                self.actor_critic,
                cfg_rl['algo']['value_loss_coef'],
                cfg_rl['algo']['entropy_coef'],
                use_recurrent_generator=cfg_rl['algo']['use_recurrent_generator'],
                writer=writer,
                acktr=True)
        elif cfg_rl['algo']['name'] == 'reinforce':
            cfg_reinforce = config['rl']['algo']['reinforce']
            self.agent = REINFORCE(
                self.actor_critic,
                cfg_reinforce['clip_param'],
                cfg_reinforce['reinforce_epoch'],
                cfg_reinforce['num_mini_batch'],
                cfg_rl['algo']['entropy_coef'],
                lr=cfg_rl['algo']['lr'],
                eps=cfg_reinforce['eps'],
                max_grad_norm=cfg_algo['max_grad_norm'],
                use_recurrent_generator=cfg_rl['algo']['use_recurrent_generator'],
                decoder_rl_loss_coef=cfg_rl['loss']['decoder_rl_loss_coef'],
                condition_rl_loss_coef=cfg_rl['loss']['condition_rl_loss_coef'],
                latent_rl_loss_coef=cfg_rl['loss']['latent_rl_loss_coef'],
                setup=config['algorithm'],
                use_mean_only=cfg_rl['loss']['use_mean_only_for_latent_loss'],
                writer=writer)

        self.rollouts = RolloutStorage2(cfg_rl['num_steps'], cfg_rl['num_processes'], self.envs.observation_space.shape,
                                        self.envs.action_space, self.recurrent_hidden_state_size,
                                        cfg_rl['policy']['execution_guided'],
                                        cfg_envs['executable']['num_demo_per_program'],
                                        cfg_envs['executable']['max_demo_length'], exec_env_state_shape,
                                        cfg_envs['executable']['dense_execution_reward'],
                                        future_rewards=cfg_rl['future_rewards'])

        obs = self.envs.reset()
        self.rollouts.obs[0].copy_(obs)
        self.rollouts.to(device)
        if config['net']['controller']['use_previous_programs']:
            self.rollouts.recurrent_hidden_states = config['net']['controller']['input_coef'] * torch.ones_like(
                self.rollouts.recurrent_hidden_states)

        self.episode_rewards = deque(maxlen=10)

        self.num_steps = cfg_rl['num_steps']
        self.num_processes = cfg_rl['num_processes']
        self.num_updates = int(cfg_rl['num_env_steps']) // cfg_rl['num_steps'] // cfg_rl['num_processes']

        final_entropy = cfg_algo['final_entropy_coef'] if cfg_algo['use_exp_ent_decay'] else cfg_algo['entropy_coef']
        self.ent_coef_sched = HyperParameterScheduler(initial_val=cfg_algo['entropy_coef'], num_updates=self.num_updates,
                                                      final_val=final_entropy, func='exponential')

        # set simple moving average of program_embedding distances
        self.d_m = torch.zeros(self.num_processes, 1).to(self.device)
        # refer to NGU paper hyperparameters
        # self.s_m = (8/10) * (self.num_steps/2)
        self.s_m = (8 / 10)

        if self.config['debug']:
            self.net._debug = defaultdict(dict)

    def _get_condition_inputs(self):
        initial_states = self.envs.render(mode='init_states')
        initial_states = np.moveaxis(initial_states, [-1,-2,-3], [-3,-2,-1])
        # B x num_demo_per_program x 1 x c x w x h
        condition_inputs = torch.tensor(initial_states).to(self.device).float().unsqueeze(2)
        return condition_inputs

    def visualization_log(self, mode, record_dict, num):
        if num < 40000:
            masks = record_dict['original_masks'][:,:5,:].detach().cpu().numpy().tolist()
            vectors = record_dict['program_latent_vectors'][:,:5,:].detach().cpu().numpy().tolist()
            dist = record_dict['distribution_params'][:,:5,:,:].detach().cpu().numpy().tolist()
            programs = record_dict['program_preds'][:,:5,:].detach().cpu().numpy().tolist()
            log_record_dict(mode,
                            {'original_masks': masks,
                             'program_latent_vectors': vectors,
                             'distribution_params': dist,
                             'program_preds': programs},
                            self.global_logs)

    def _get_condition_policy_reward(self, agent_actions, agent_action_masks, infos):
        # agent_actions: batch_size x num_demp_per_program x max_demo_len-1
        gt_a_h_all = []
        gt_a_h_len_all = []
        for i, info in enumerate(infos):
            gt_a_h = info['exec_data']['gt_a_h']
            gt_a_h_len = info['exec_data']['gt_a_h_len']
            for j, a_h in enumerate(gt_a_h):
                if gt_a_h_len[j] == 0:
                    gt_a_h_len[j] += 1
                    gt_a_h[j][0] = self.config['dsl']['num_agent_actions'] - 1
            gt_a_h_all += [torch.tensor(gt_a_h[i], device=agent_actions.device) for i in range(gt_a_h.shape[0])]
            gt_a_h_len_all += gt_a_h_len.tolist()

        packed_gt_action_seq = torch.nn.utils.rnn.pack_sequence(gt_a_h_all, enforce_sorted=False)
        padded_gt_action_seq, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_gt_action_seq, batch_first=True,
                                                                         padding_value=self.config['dsl'][
                                                                                           'num_agent_actions'] - 1,
                                                                         total_length=agent_actions.shape[-1])
        padded_gt_action_seq = padded_gt_action_seq.view(agent_actions.shape[0], agent_actions.shape[1], -1)

        gt_action_masks = padded_gt_action_seq != self.config['dsl']['num_agent_actions']-1
        action_masks = torch.max(agent_action_masks, gt_action_masks)
        condition_reward = utils.masked_mean(padded_gt_action_seq == agent_actions, action_masks, dim=-1, keepdim=True)
        return condition_reward

    def _add_debug_info(self, debug_data, location, update, step):
        current_update = 'u_{}_s_{}'.format(update, step)
        self.net._debug[location][current_update] = {}
        for key, val in debug_data.items():
            self.net._debug[location][current_update][key] = val

        keys = ['u_{}_s_{}_{}'.format(update, step, i) for i in range(debug_data['pred_programs'].shape[0])]
        self.net._debug[location][current_update]['ids'] = keys
        return current_update

    @staticmethod
    def temp_env_step(data):
        config, gt_program, pred_program, init_states = data
        env = ProgramEnv1(config, gt_program, init_states)
        env._max_episode_steps = config.max_episode_steps
        env.seed(config.seed)
        _, reward, done, info = env.step(pred_program)
        return (reward, done, info)

    def _supervisedRL_env_step(self, pred_programs, gt_programs, init_states):
        _, reward, done, infos = self.decoder_envs.step(
            torch.cat((gt_programs.long(), pred_programs), dim=-1).detach().cpu())

        return None, reward, done, infos

    def _get_latent_exploration_reward(self, program_embeddings):
        k = self.num_steps//2
        l2_norm = torch.norm(self.rollouts.program_embeddings - program_embeddings.unsqueeze(0), p=2, dim=-1)
        l2_norm = l2_norm.permute(1,0)
        d_k, d_k_idx = torch.topk(l2_norm, k, dim=-1, largest=False)

        # update moving average d_m**2
        current_avg_dist = torch.mean(d_k**2, dim=-1, keepdim=True)
        self.d_m = (self.d_m + current_avg_dist) / 2

        # normalize the distances with moving average
        d_n = d_k / self.d_m
        d_n = torch.clamp(d_n - 0.008, min=0)

        # compute the kernel
        K = 0.0001 / (d_n + 0.0001)
        s = torch.sqrt(torch.sum(K, dim=-1)) + 0.001

        s = s.detach().cpu().view(-1,1)
        return s

    def update_step(self, j, start, config, mode='train'):
        cfg_rl = config['rl']
        cfg_algo = config['rl']['algo']
        cfg_envs = config['rl']['envs']
        use_decoder_dist = config['net']['controller']['use_decoder_dist']

        t = time.time()
        if cfg_algo['use_linear_lr_decay']:
            # decrease learning rate linearly
            current_learning_rate = utils.update_linear_schedule(self.agent.optimizer, j, self.num_updates,
                                                                 cfg_algo['lr'])
        else:
            current_learning_rate = cfg_algo['lr']

        for step in range(self.num_steps):
            if config['algorithm'] != 'supervisedRL':
                condition_inputs = self._get_condition_inputs()
            else:
                condition_inputs = self.rollouts.agent_initial_states[step]

            # Sample actions
            with torch.no_grad():
                outputs = self.actor_critic.act(self.rollouts.obs[step],
                                                self.rollouts.recurrent_hidden_states[step],
                                                self.rollouts.masks[step],
                                                condition_inputs, deterministic=not use_decoder_dist)
                value, pred_programs, pred_programs_log_probs, recurrent_hidden_states, pred_program_masks, \
                eop_pred_programs, agent_value, agent_actions, agent_action_log_probs, agent_action_masks, \
                program_embeddings, distribution_params, _, _, latent_log_probs, _ = outputs

                """program_env expects EOP token in action but two_head policy can't have it as part of action space
                   so we manually add EOP token for two_head_policy over here"""
                if cfg_envs['program']['mdp_type'] == 'ProgramEnv1':
                    alt_pred_programs = (self.actor_critic.num_program_tokens - 1) * torch.ones_like(pred_programs)
                    pred_programs2 = torch.where(pred_program_masks > 0, pred_programs, alt_pred_programs) if \
                    cfg_rl['policy']['two_head'] else pred_programs
                else:
                    pred_programs2 = pred_programs.clone()

            # add debug info
            if self.config['debug']:
                debug_id = self._add_debug_info({'pred_programs': pred_programs,
                                                 'pred_program_log_probs': pred_programs_log_probs,
                                                 'pred_program_masks': pred_program_masks,
                                                 'agent_actions': agent_actions,
                                                 'agent_action_log_probs': agent_action_log_probs,
                                                 'agent_action_masks': agent_action_masks,
                                                 'program_embeddings': program_embeddings,
                                                 'distribution_params': distribution_params},
                                                location='act',
                                                update=j,
                                                step=step)

            # Observation reward and next obs
            if self.config['algorithm'] != 'supervisedRL':
                obs, reward, done, infos = self.envs.step(pred_programs2)
            else:
                _, reward, done, infos = self._supervisedRL_env_step(pred_programs2, self.rollouts.obs[step],
                                                                       condition_inputs)
                obs = self.rollouts.obs[step + 1]

            # Add latent space exploration reward based on current memory
            reward_intrinsic = None
            reward_env = reward
            if cfg_envs['program']['intrinsic_reward']:
                beta = cfg_envs['program']['intrinsic_beta']
                s = self._get_latent_exploration_reward(program_embeddings)
                reward_intrinsic = 1/s
                reward = torch.where(s > self.s_m, reward, reward + beta * reward_intrinsic)

            if config['rl']['loss']['condition_rl_loss_coef'] > 0.0:
                if cfg_envs['executable']['task_definition'] == 'program':
                    agent_action_done = done
                    agent_action_reward = self._get_condition_policy_reward(agent_actions, agent_action_masks, infos)
                else:
                    agent_action_obs, agent_action_reward, agent_action_done, agent_action_info = self.condition_envs.step(
                        agent_actions)
            else:
                agent_action_reward = torch.zeros(
                    (self.num_processes, cfg_envs['executable']['num_demo_per_program'], 1),
                    device=agent_actions.device)

            # FIXME: There is no episode info in infos for supervisedRL
            for i, info in enumerate(infos):
                if 'episode' in info.keys():
                    assert done[i]
                    self.episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if (done_ and config['rl']['use_all_programs']) else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            orig_masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            debug_data = self.net._debug['act'][debug_id] if self.config['debug'] else None
            self.rollouts.insert(obs, recurrent_hidden_states, pred_programs, pred_programs_log_probs, value,
                                 reward, reward_intrinsic, reward_env, masks, bad_masks, pred_program_masks,
                                 eop_pred_programs, agent_value, agent_actions, agent_action_log_probs,
                                 agent_action_reward, agent_action_masks, program_embeddings, condition_inputs,
                                 distribution_params, orig_masks, latent_log_probs, debug_data)

        # logging
        cum_reward = self.rollouts.rewards.cpu().numpy()
        self.writer.add_scalar('agent/reward_sum', cum_reward.sum(), j)
        self.writer.add_scalar('agent/reward_max', cum_reward.max(), j)
        self.writer.add_scalar('agent/reward_min', cum_reward.min(), j)
        self.writer.add_scalar('agent/reward_mean', cum_reward.mean(), j)
        self.writer.add_scalar('agent/reward_intrinsic_mean', self.rollouts.rewards_intrinsic.cpu().numpy().mean(), j)
        self.writer.add_scalar('agent/reward_env_sum', self.rollouts.rewards_env.cpu().numpy().sum(), j)
        self.writer.add_scalar('agent/reward_env_mean', self.rollouts.rewards_env.cpu().numpy().mean(), j)
        self.writer.add_scalar('agent/reward_env_max', self.rollouts.rewards_env.cpu().numpy().max(), j)
        self.writer.add_scalar('agent/reward_env_min', self.rollouts.rewards_env.cpu().numpy().min(), j)
        self.writer.add_scalar('agent/learning_rate', current_learning_rate, j)
        self.writer.add_scalar('distribution/mean',
                               torch.norm(self.rollouts.program_distribution_params[:, 0, :].detach(), dim=-1,
                                          p=2).mean().cpu().numpy(), j)
        self.writer.add_scalar('distribution/std',
                               torch.norm(self.rollouts.program_distribution_params[:, 1, :].detach(), dim=-1,
                                          p=2).mean().cpu().numpy(), j)

        # FIXME: I don't think we need this if check anymore, but somebody gotta verify it
        if len(pred_programs.shape) == 1:
            pred_programs = pred_programs.reshape(1, -1)
        # pred program str
        if "karel" or "CartPoleDiscrete-v0" in cfg_envs['executable']['name']:
            if cfg_envs['program']['mdp_type'] == 'ProgramEnv1':
                pred_program_strs = [self.dsl.intseq2str(info['modified_action']) for info in infos]
                num_tokens = self.actor_critic.num_program_tokens
                if cfg_rl['policy']['two_head']:
                    program_len = torch.sum(rollouts.pred_program_masks.view(-1, config['dsl']['max_program_len']),
                                            dim=1).float().cpu().numpy()
                else:
                    program_len = config['dsl']['max_program_len'] - torch.sum(
                        self.rollouts.actions.view(-1, config['dsl']['max_program_len']) == num_tokens - 1,
                        dim=1).float().cpu().numpy()
                self.writer.add_scalar('agent/mean_program_len', program_len.mean(), j)
                for i, pred_program_str in enumerate(pred_program_strs):
                    self.writer.add_text('program/pred_{}'.format(i),
                                         'reward_env: {} program: {} '.format(reward_env[i].cpu().numpy(),
                                                                          pred_program_str), j)
            else:
                raise NotImplementedError('not yet implemented!')

        gamma = cfg_rl['gamma'] if cfg_rl['future_rewards'] else 0
        if cfg_algo['name'] == 'reinforce':
            self.rollouts.compute_returns(0, 0, False, gamma, None, False, cfg_algo['name'])
        else:
            with torch.no_grad():
                next_value, agent_next_value = self.actor_critic.get_value(
                    self.rollouts.obs[-1], self.rollouts.recurrent_hidden_states[-1],
                    self.rollouts.masks[-1], self.rollouts.agent_initial_states[-1],
                    deterministic=not use_decoder_dist)

            self.rollouts.compute_returns(next_value, agent_next_value, cfg_rl['use_gae'], gamma, cfg_rl['gae_lambda'],
                                          cfg_rl['use_proper_time_limits'], cfg_algo['name'])

        # RL algorithm update
        value_loss, action_loss, dist_entropy = self.agent.update(self.rollouts, use_decoder_dist=use_decoder_dist)

        # log all items in dict
        record_dict = {'udpate_num': j, 'update_action_loss': action_loss,
                       'update_entropy': dist_entropy,
                       'update_reward_mean': self.rollouts.rewards.cpu().numpy().mean(),
                       'update_reward_sum': self.rollouts.rewards.cpu().numpy().sum(),
                       'update_reward_min': self.rollouts.rewards.cpu().numpy().min(),
                       'update_reward_max': self.rollouts.rewards.cpu().numpy().max(),
                       'update_reward_env_mean': self.rollouts.rewards_env.cpu().numpy().mean(),
                       'update_reward_env_sum': self.rollouts.rewards_env.cpu().numpy().sum(),
                       'update_reward_env_min': self.rollouts.rewards_env.cpu().numpy().min(),
                       'update_reward_env_max': self.rollouts.rewards_env.cpu().numpy().max(),
                       }
        log_record_dict(mode, record_dict, self.global_logs)
        if self.config['algorithm'] == 'RL':
            self.visualization_log(mode, {'original_masks': self.rollouts.original_masks,
                                             'program_latent_vectors': self.rollouts.program_embeddings,
                                             'distribution_params': self.rollouts.program_distribution_params,
                                             'program_preds': self.rollouts.actions}, j * self.num_steps)
        if self.verbose:
            self._print_record_dict(record_dict, j, self.num_updates, 'train', time.time() - t)

        self.writer.add_scalar('agent/mean_program_value_estimate_2', self.rollouts.value_preds.cpu().numpy().mean(), j)
        self.logger.info("PPO value_loss: {} policy_loss: {} dist_entropy: {}".format(value_loss, action_loss, dist_entropy))

        self.rollouts.after_update()
        self.agent.entropy_coef = self.ent_coef_sched.step(j)

        # save for every interval-th episode or for the last epoch
        if (j % config['save_interval'] == 0
            or j == self.num_updates - 1) and config['outdir'] != "":
            save_path = os.path.join(config['outdir'], cfg_algo['name'])
            create_directory(save_path)
            self.save_net(os.path.join(save_path, cfg_envs['executable']['name'] + '_' + str(j) + ".pt"))

        if j % config['log_interval'] == 0 and len(self.episode_rewards) > 1:
            total_num_steps = (j + 1) * self.num_processes * self.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, "
                "min/max reward {:.1f}/{:.1f}\n "
                    .format(j, total_num_steps,
                            int(total_num_steps / (end - start)),
                            len(self.episode_rewards), np.mean(self.episode_rewards),
                            np.median(self.episode_rewards), np.min(self.episode_rewards),
                            np.max(self.episode_rewards), dist_entropy, value_loss,
                            action_loss))

        # Save results
        if j % (config['save_interval'] // 10) == 0:
            pickle.dump(self.global_logs,
                        file=open(os.path.join(self.config['outdir'], self.config['record_file']), 'wb'))

        # FIXME: add evaluation code as in main repo

        return None

    def train(self,  *args, **kwargs):
        start = time.time()
        config = self.config
        for j in range(self.num_updates):
            self.update_step(j, start, config)

        return None

    def save_net(self, filename):
        params = [self.actor_critic.state_dict(), getattr(utils.get_vec_normalize(self.envs), 'ob_rms', None)]
        torch.save(params, filename)
        self.logger.debug('params saved to {}'.format(filename))

    def load_net(self, filename):
        self.logger.debug('Loading params from {}'.format(filename))
        params = torch.load(filename, map_location=self.device)
        self.actor_critic.load_state_dict(params[0], strict=False)

    def load_net2(self, filename):
        self.logger.debug('Loading params from {}'.format(filename))
        params = torch.load(filename, map_location=self.device)
        self.actor_critic.program_vae.load_state_dict(params[0])

    def _print_record_dict(self, record_dict, current_update, total_updates, usage, t_taken):
        loss_str = ''
        for k, v in record_dict.items():
            loss_str = loss_str + ' {}: {:.8f}'.format(k, v)

        loss_str = 'update {}/{}: '.format(current_update, total_updates) + loss_str
        self.logger.debug('{}:{} took {:.3f}s'.format(usage, loss_str, t_taken))
        return None
