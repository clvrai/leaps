import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
        id='CartPoleDiscrete-v0',
        entry_point='prl_gym.envs:CartPoleDiscreteEnv',
)
