import sys
sys.path.insert(0, 'karel_env/dsl')
from dsl_prob import DSLProb


def get_DSL(dsl_type='prob', seed=None, environment='karel'):
    if dsl_type == 'prob':
        return DSLProb(seed=seed, environment=environment)
    else:
        raise ValueError('Undefined dsl type')