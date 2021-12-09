"""
NOTE:
     This file works only on Karel DSL regardless of whether we use simplified DSL for training or not.
     So if any of the public functions change, make sure that you call _prl_to_dsl function appropriately
     as we do in ProgramEnv.step (via _modify())
"""
import time
import numpy as np
import gym
from gym import spaces
from exec_env import ExecEnv1, ExecEnv2
from karel_env.dsl.dsl_parse import parse
from fetch_mapping import fetch_mapping


class ProgramEnv(gym.Env):
    """Environment that will follow gym interface"""

    def __init__(self, config, task=None, metadata={}):
        super(ProgramEnv, self).__init__()
        self.metadata = {'render.modes': ['rgb_array', 'program', 'init_states']}
        self.config = config
        self.max_program_len = config.max_program_len

        # load task definition (task can be defined by program or inbuilt environment reward function)
        if self.config.task_definition == 'program':
            self.task_env = ExecEnv1(config, task, metadata)
            self.gt_reward, _ = self.task_env.reward(self.task_env.gt_program_seq)
        elif self.config.task_definition == 'custom_reward':
            self.gt_reward = 10000.0
            self.task_env = ExecEnv2(config, metadata)
        else:
            raise NotImplementedError

        # Add one token for invalid token (all tokens after end token should be invalid)
        if config.use_simplified_dsl:
            self.num_program_tokens = len(config.prl_tokens)+1
            self.T2I = {tkn: i for i, tkn in enumerate(config.prl_tokens)}
        else:
            self.num_program_tokens = len(self.task_env.dsl.int2token)+1
            self.T2I = {tkn: i for i, tkn in enumerate(config.dsl_tokens)}

        self._elapsed_steps = 0
        self.partial_program = []

    def _prl_to_dsl(self, program_seq):
        def func(x):
            return self.config.dsl_tokens.index(self.config.prl2dsl_mapping[self.config.prl_tokens[x]])
        return np.array(list(map(func, program_seq)), program_seq.dtype)

    def _set_bad_transition(self, done, info):
        # TODO: need to shift this code under rl.envs.TimeLimitMask
        if self._elapsed_steps >= self._max_episode_steps:
            info['TimeLimit.truncated'] = not done
            info['bad_transition'] = done
            done = True
        return done, info

    def step(self, action):
        raise NotImplementedError()

    def render(self, mode='init_states'):
        """render current program for a random initial state"""
        if mode == 'program':
            pred_program = self.task_env.execute_pred_program(self.state)
            return pred_program
        elif mode == 'init_states':
            return self.task_env.render(mode='init_states')
        else:
            raise NotImplementedError('Yet to generate video of predicted program execution')


class ProgramEnv1(ProgramEnv):
    """MDP2
        state: None
        action: complete program
        Transition: complete program -> complete program
        reward: environment reward for executing the program
    """

    def __init__(self, config, task=None, metadata={}):
        super(ProgramEnv1, self).__init__(config, task, metadata)

        # define action space
        self.alpha = metadata.get('alpha', 1)
        if config.action_type == "program_multidiscrete":
            self.action_space = spaces.MultiDiscrete(self.alpha*self.max_program_len*[self.num_program_tokens])
        elif config.action_type == "program":
            self.action_space = spaces.Box(low=0, high=self.num_program_tokens,
                                           shape=(self.alpha*self.max_program_len,), dtype=np.int8)

        # define observation space
        if config.obv_type == "program":
            self.observation_space = spaces.Box(low=0, high=self.num_program_tokens,
                                                shape=(self.max_program_len,), dtype=np.int8)
            self.initial_obv = (self.num_program_tokens-1) * np.ones(self.max_program_len, dtype=np.int8)
        elif config.obv_type == "encoded":
            self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'),
                                                shape=[config.num_lstm_cell_units], dtype=np.float32)
            self.initial_obv = np.zeros(config.num_lstm_cell_units)
        else:
            raise NotImplementedError('observation not recognized')

        self.state = self.initial_obv

    def _modify(self, action):
        # Ignore everything after end-of-program token
        null_token_idx = np.argwhere(action == (self.num_program_tokens-1))
        if null_token_idx.shape[0] > 0:
            action = action[:null_token_idx[0].squeeze()]
        # remap prl tokens to dsl tokens if we are using simplified DSL
        action = self._prl_to_dsl(action) if self.config.use_simplified_dsl else action
        return action

    def step(self, action):
        """Currently state is previous program, action is new program
        Alert: action can be in simplified DSL format, make sure to use transformed action
               (here we transform it in _modify())
        """
        if self.alpha > 1:
            gt_program_seq, pred_program_seq = action[:len(action)//2], action[len(action)//2:]
            self.task_env.gt_program_seq = gt_program_seq
            self.task_env.gt_program = self.task_env._execute_gt_program(self.config, gt_program_seq)
            action = pred_program_seq
        self._elapsed_steps += 1
        dsl_action = self._modify(action)
        # FIXME: temporary fix for ignoring DEF, run, )m kind of tokens
        if self.config.experiment == 'intention_space':
            dsl_action = np.concatenate((np.array([0]), dsl_action))
        else:
            if self.config.grammar is not None:
                dsl_action = np.concatenate((np.array([0, 1, 2]), dsl_action))
            else:
                dsl_action = np.concatenate((np.array([0, 1, 2]), dsl_action, np.array([3])))

        self.state = program_seq = dsl_action
        reward, exec_data = self.task_env.reward(program_seq)
        done = True if reward == self.gt_reward else False
        info = {'cur_state': action, 'modified_action': dsl_action, 'exec_data': exec_data}

        done, info = self._set_bad_transition(done, info)

        return self.initial_obv, reward, done, info

    def reset(self):
        # Reset the state of the environment to an initial state
        self._elapsed_steps = 0
        self.partial_program = []
        self.state = self.initial_obv
        self.task_env.reset()
        return self.state
