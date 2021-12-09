import multiprocessing
import numpy as np
import torch
import torch.nn as nn
import os
import glob
import pickle
from collections import OrderedDict, deque
from operator import itemgetter
from abc import ABCMeta, abstractmethod

from pretrain.models import ProgramVAE
from rl.envs import make_vec_envs
from rl import utils
from utils.misc_utils import HyperParameterScheduler


class CrossEntropyNet(nn.Module):
    def __init__(self, envs, config):
        super(CrossEntropyNet, self).__init__()
        self.vector_init_type = config['CEM']['init_type']
        self.program_vae = ProgramVAE(envs, **config)

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.program_vae.vae.latent_dim

    def get_init_vector(self, size, device):
        if self.vector_init_type == 'zeros':
            return torch.zeros(size, device=device)
        elif self.vector_init_type == 'ones':
            return 0.01 * torch.ones(size, device=device)
        elif self.vector_init_type == 'normal':
            return torch.randn(size, device=device)
        elif self.vector_init_type == 'tiny_normal':
            return torch.normal(0.0, 0.1, size=(size,), device=device)
        elif self.vector_init_type == 'large_normal':
            return torch.randn(0.0, 2.0, size=(size,), device=device)
        else:
            raise NotImplementedError()

    def forward(self, h_meta, deterministic=False):
        z = h_meta
        """ decode sampled latent vector to a program """
        output = self.program_vae.vae.decoder(None, z, teacher_enforcing=False, deterministic=deterministic,
                                              evaluate=False)

        _, pred_programs, pred_programs_len, pred_programs_log_probs, output_logits, eop_pred_programs,\
        eop_output_logits, pred_program_masks, dist_entropy = output
        return pred_programs


class CrossEntropyAgent(object):
    """Agent that uses Cross Entropy to learn"""
    def __init__(self, device, logger, config, envs):
        self.device = device
        self.logger = logger
        self.config = config
        self.model = CrossEntropyNet(envs, config)
        self.model.to(device)
        checkpt = self.config['net']['saved_params_path']
        if checkpt is not None:
            self.logger.debug('Loading params from {}'.format(checkpt))
            params = torch.load(checkpt, map_location=self.device)
            self.model.program_vae.load_state_dict(params[0], strict=False)

        self._best_vector = self.model.get_init_vector(config['num_lstm_cell_units'], device)
        self._best_score = 0
        self._best_program = None
        self._best_program_str = ""

        self.reduction = config['CEM']['reduction']

        # self.workers = [self.create_network(config) for _ in range(config["population_size"])]

        self.n_elite = round(config["CEM"]["population_size"] * config["CEM"]["elitism_rate"])

        self.final_sigma = config['CEM']['final_sigma'] if config['CEM']['use_exp_sig_decay'] else config['CEM']['sigma']
        self.sigma_sched = HyperParameterScheduler(initial_val=config['CEM']['sigma'],
                                                      num_updates=config['CEM']['max_number_of_epochs']/2,
                                                      final_val=self.final_sigma, func='exponential')
        self.current_sigma = config['CEM']['sigma']

    def act(self, state, deterministic=False):
        """select one action based on the current state"""
        return self.model(state, deterministic=deterministic)

    def learn(self, envs, best_env):
        """run one learning step"""
        results = {}
        current_population = [self.best_vector + (self.current_sigma * torch.randn_like(self.best_vector)) for
                              _ in range(self.config['CEM']['population_size'])]
        current_population = torch.stack(current_population, dim=0)
        with torch.no_grad():
            pred_programs = self.act(current_population)

        obs, reward, done, infos = envs.step(pred_programs)
        if self.config['CEM']['exponential_reward']:
            reward = torch.exp(reward)
        for i, info in enumerate(infos):
            results[i] = (reward[i].squeeze().detach().cpu().numpy(), info['exec_data']['program_prediction'])

        sorted_results = OrderedDict(sorted(results.items(), key=itemgetter(1)))
        elite_idxs = list(sorted_results.keys())[-self.n_elite:]

        if self.reduction == 'mean':
            self._best_vector = torch.mean(current_population[elite_idxs], dim=0)
        elif self.reduction == 'max':
            self._best_vector, _ = torch.max(current_population[elite_idxs], dim=0)
        elif self.reduction == 'weighted_mean':
            reward = reward.to(self.device)
            self._best_vector = torch.sum(reward[elite_idxs] * current_population[elite_idxs], dim=0) / (torch.sum(reward[elite_idxs]) + 1e-5)
            #warnings.warn("Warning...........weighted_mean method is chosen for CEM aggregation, make sure to define"
            #              " weighted mean based on calculated rewards")
        with torch.no_grad():
            self._best_program = self.act(torch.stack((self.best_vector, self.best_vector)), deterministic=True)[0]

        _, best_reward, _, best_infos  = best_env.step(self.best_program.unsqueeze(0))
        if self.config['CEM']['exponential_reward']:
            best_reward = torch.exp(best_reward)
        self._best_score = best_reward.detach().cpu().numpy()
        self._best_program_str = best_infos[0]['exec_data']['program_prediction']

        return results, pred_programs[elite_idxs].detach().cpu().numpy(),\
               current_population[elite_idxs].detach().cpu().numpy(), self._best_score

        # worker_results = self._run_worker(envs)
        # sorted_results = OrderedDict(sorted(worker_results.items(), key=itemgetter(1)))
        # elite_idxs = list(sorted_results.keys())[-self.n_elite:]
        #
        # elite_weighs = [self.workers[i].get_weights() for i in elite_idxs]
        # self.best_weights = [np.array(weights).mean(axis=0) for weights in zip(*elite_weighs)]
        # self.model.set_weights(self.best_weights)
        # self.best_score = self.objective_function(self.model, gym.make(self.config["env_name"]),
        #                                           self.config["max_steps_in_episodes"], 1.0)
        # return self.best_score

    @property
    def best_program(self):
        return self._best_program

    @property
    def best_program_str(self):
        return self._best_program_str

    @property
    def best_vector(self):
        return self._best_vector

    @property
    def best_score(self):
        return self._best_score

class CEMModel(object):
    def __init__(self, device, config, dummy_envs, dsl, logger, writer, global_logs, verbose):
        self.device = device
        self.config = config
        self.global_logs = global_logs
        self.verbose = verbose
        self.logger = logger
        self.writer = writer
        self.envs = dummy_envs
        self.dsl = dsl

        ##########################################
        log_dir = os.path.expanduser(os.path.join(config['outdir'], 'CEM', 'openai_CEM'))
        log_dir_best = os.path.expanduser(os.path.join(config['outdir'], 'CEM', 'openai_CEM','best'))
        eval_log_dir = log_dir + "_eval"
        utils.cleanup_log_dir(log_dir)
        utils.cleanup_log_dir(log_dir_best)
        utils.cleanup_log_dir(eval_log_dir)

        cfg_rl = config['rl']
        cfg_envs = config['rl']['envs']

        custom = True if "karel" or "CartPoleDiscrete" in cfg_envs['executable']['name'] else False
        logger.info('Using environment: {}'.format(cfg_envs['executable']['name']))
        self.envs = make_vec_envs(cfg_envs['executable']['name'], config['seed'], config['CEM']['population_size'],
                                  cfg_rl['gamma'], os.path.join(config['outdir'], 'CEM', 'openai_CEM'), device, False,
                                  custom_env=custom, custom_env_type='program', custom_kwargs={'config': config['args']})
        obs = self.envs.reset()

        self.best_env = make_vec_envs(cfg_envs['executable']['name'], config['seed'], 1, cfg_rl['gamma'],
                                      os.path.join(config['outdir'], 'CEM', 'openai_CEM', 'best'), device, False,
                                      custom_env=custom, custom_env_type='program',
                                      custom_kwargs={'config': config['args']})
        self.best_env.reset()


        self.agent = CrossEntropyAgent(device, logger, config, self.envs)

        self.gt_program_str = open(cfg_envs['executable']['task_file']).readlines()[0].strip()

    def train(self):
        """run max_number_of_episodes learning epochs"""
        self.writer.add_text('program/ground_truth', 'program: {} '.format(self.gt_program_str), 0)
        scores_deque = deque(maxlen=10)
        scores = []
        for epoch in range(1, self.config['CEM']['max_number_of_epochs'] + 1):
            results, elite_programs, elite_vectors, reward = self.agent.learn(self.envs, self.best_env)
            self.agent.current_sigma = self.agent.sigma_sched.step(
                epoch - 1) if self.agent.sigma_sched.cur_step <= self.agent.sigma_sched.total_num_epoch else self.agent.final_sigma
            scores.append(reward)
            scores_deque.append(reward)

            if np.mean(scores_deque) >= self.config['CEM']['average_score_for_solving'] and self.agent.best_score >= \
                    self.config['CEM']['average_score_for_solving'] and len(scores_deque) >= 10:
                self.logger.debug("\nEnvironment solved after episode: {}".format(epoch - 10))
                self.logger.debug("\nMean Reward over {} episodes: {}".format(epoch, np.mean(scores_deque)))
                self.save(os.path.join(self.config["outdir"], 'CEM', 'final_vectors.pkl'), converged=True)
                break

            if epoch % self.config['save_interval'] == 0:
                self.save(os.path.join(self.config["outdir"], 'CEM', str(epoch)+'_vectors.pkl'))
            self.save(os.path.join(self.config["outdir"], 'CEM', 'final_vectors.pkl'))
            print_str = "Episode: {} - current mean Reward: {} best reward: {} best program: {}".format(epoch, np.mean(
                scores_deque), self.agent.best_score, self.agent.best_program_str)
            self.logger.debug(print_str)

            # Add logs to TB
            self.writer.add_scalar('agent/best_reward', self.agent.best_score, epoch)
            self.writer.add_scalar('agent/avg_score_queue', np.mean(scores_deque), epoch)
            self.writer.add_text('program/best_{}'.format(epoch),
                                 'reward_env: {} program: {} '.format(self.agent.best_score,
                                                                      self.agent.best_program_str), epoch)
            print(print_str)


        return scores

    def save(self, filename, converged=False):
        """Save the network weights"""
        with open(filename, 'wb') as f:
            best_program = {'vector': self.agent.best_vector, 'program_str': self.agent.best_program_str,
                            'reward': self.agent.best_score, 'converged': converged}
            pickle.dump(best_program, f)

    def load(self):
        """Load latest available network weights"""
        filename = os.path.join(self.config["outdir"], 'CEM', "final_weights.pkl")
        with open(filename, "rb") as f:
            self.agent._best_vector = pickle.load(f)
