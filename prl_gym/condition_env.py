import gym
from gym import spaces

class ConditionEnvGym(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, num_agent_actions, max_demo_length):
        super(ConditionEnvGym, self).__init__()

        self.config = config
        if self.config.env_name == "karel":
            self.dsl = get_DSL(dsl_type='prob', seed=config.seed, environment=self.config.env_name)
            self.s_gen = KarelStateGenerator(seed=config.seed)
            self._world = karel.Karel_world(make_error=False, env_task=config.env_task, reward_diff=config.reward_diff)
        elif self.config.env_name == "CartPoleDiscrete-v0":
            self.dsl = get_DSL(dsl_type='prob', seed=config.seed, environment=self.config.env_name)
            gym_env = gym.make(self.config.env_name)
            self._world = cartpole.CartPole_World(gym_env)
        else:
            raise NotImplementedError('{} not implemented for PRL setup'.format(self.config.env_name))

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Box(low=0, high=num_agent_actions+1,
                                           shape=(max_demo_length,), dtype=np.int16)
        # Example for using image as input:
        self.observation_space = spaces.Box(low=0, high=1, shape=(height, width, 16), dtype=np.bool_)

        if env_task == 'maze':
            self.init_func = self.s_gen.generate_single_state_find_marker
        elif env_task == 'stairClimber':
            self.init_func = self.s_gen.generate_single_state_stair_climber
        else:
            raise NotImplementedError('task not implemented yet')

        self.init_states = [self.init_func(height, width) for _ in range(config.num_demo_per_program)]
        self._world.set_new_state(self.init_states[0][0])

    def step(self, action):
        return observation, reward, done, info

    def reset(self):

        return observation

    def render(self, mode='init_states'):
        if mode == 'init_states':
            return [x[0] for x in self.init_states]
        else:
            return self._world.render(mode)

    def close (self):
        raise NotImplementedError()