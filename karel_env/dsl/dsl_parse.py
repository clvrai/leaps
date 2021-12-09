import numpy as np
from collections import defaultdict

import dsl_data


def check_and_apply(queue, rule):
    r = rule[0].split()
    l = len(r)
    if len(queue) >= l:
        t = queue[-l:]
        if list(list(zip(*t))[0]) == r:
            new_t = rule[1](list(list(zip(*t))[1]))
            del queue[-l:]
            queue.extend(new_t)
            return True
    return False

rules = []

# k, n, s = fn(k, n)
# k: karel_world
# n: num_call
# s: success
# c: condition [True, False]
MAX_FUNC_CALL = 220


def r_prog(t):
    stmt = t[3]

    def fn(k, n):
        if n > MAX_FUNC_CALL: return k, n, False
        return stmt(k, n + 1)
    return [('prog', fn)]
rules.append(('DEF run m( stmt m)', r_prog))


def r_stmt(t):
    stmt = t[0]

    def fn(k, n):
        if n > MAX_FUNC_CALL: return k, n, False
        return stmt(k, n + 1)
    return [('stmt', fn)]
rules.append(('while_stmt', r_stmt))
rules.append(('repeat_stmt', r_stmt))
rules.append(('stmt_stmt', r_stmt))
rules.append(('action', r_stmt))
rules.append(('if_stmt', r_stmt))
rules.append(('ifelse_stmt', r_stmt))


def r_stmt_stmt(t):
    stmt1, stmt2 = t[0], t[1]

    def fn(k, n):
        if n > MAX_FUNC_CALL: return k, n, False
        k, n, s = stmt1(k, n + 1)
        if not s: return k, n, s
        if n > MAX_FUNC_CALL: return k, n, False
        return stmt2(k, n)
    return [('stmt_stmt', fn)]
rules.append(('stmt stmt', r_stmt_stmt))


def r_if(t):
    cond, stmt = t[2], t[5]

    def fn(k, n):
        if n > MAX_FUNC_CALL: return k, n, False
        k, n, s, c = cond(k, n + 1)
        if not s: return k, n, s
        if c: return stmt(k, n)
        else: return k, n, s
    return [('if_stmt', fn)]
rules.append(('IF c( cond c) i( stmt i)', r_if))


def r_ifelse(t):
    cond, stmt1, stmt2 = t[2], t[5], t[9]

    def fn(k, n):
        if n > MAX_FUNC_CALL: return k, n, False
        k, n, s, c = cond(k, n + 1)
        if not s: return k, n, s
        if c: return stmt1(k, n)
        else: return stmt2(k, n)
    return [('ifelse_stmt', fn)]
rules.append(('IFELSE c( cond c) i( stmt i) ELSE e( stmt e)', r_ifelse))


def r_while(t):
    cond, stmt = t[2], t[5]

    def fn(k, n):
        if n > MAX_FUNC_CALL: return k, n, False
        k, n, s, c = cond(k, n + 1)
        if not s: return k, n, s
        while(c):
            k, n, s = stmt(k, n)
            if not s: return k, n, s
            k, n, s, c = cond(k, n)
            if not s: return k, n, s
        return k, n, s
    return [('while_stmt', fn)]
rules.append(('WHILE c( cond c) w( stmt w)', r_while))


def r_repeat(t):
    cste, stmt = t[1], t[3]

    def fn(k, n):
        if n > MAX_FUNC_CALL: return k, n, False
        n += 1
        s = True
        for _ in range(cste()):
            k, n, s = stmt(k, n)
            if not s: return k, n, s
        return k, n, s
    return [('repeat_stmt', fn)]
rules.append(('REPEAT cste r( stmt r)', r_repeat))


def r_cond1(t):
    cond = t[0]

    def fn(k, n):
        if n > MAX_FUNC_CALL: return k, n, False, False
        return cond(k, n)
    return [('cond', fn)]
rules.append(('cond_without_not', r_cond1))


def r_cond2(t):
    cond = t[2]

    def fn(k, n):
        if n > MAX_FUNC_CALL: return k, n, False, False
        k, n, s, c = cond(k, n)
        return k, n, s, not c
    return [('cond', fn)]
rules.append(('not c( cond c)', r_cond2))


env_rules = defaultdict(list)
for env in dsl_data.envs:
    # Condition tokens
    func_str = '''
def {}_r_cond_without_not_{}(t):
    def fn(k, n):
        if n > MAX_FUNC_CALL: return k, n, False
        c = k.{}()
        return k, n, True, c
    return [('cond_without_not', fn)]
    '''

    for token, api in dsl_data.obv_token_api_dict[env].items():
        current_func_str = func_str.format(env.replace('-','_'), token, api)
        exec(current_func_str)
        fn = eval('{}_r_cond_without_not_{}'.format(env.replace('-','_'), token))
        env_rules[env].append((token, fn))

    # Action tokens
    func_str = '''
def {}_r_action{}(t):
    def fn(k, n):
        if n > MAX_FUNC_CALL: return k, n, False
        action = np.zeros({})
        action[{}] = 1
        try: k.state_transition(action)
        except: raise RuntimeError()
        #except: return k, n, False
        else: return k, n, True
    return [('action', fn)]
    '''

    for i, token in enumerate(dsl_data.action_token_list[env]):
        current_func_str = func_str.format(env.replace('-','_'), i+1, len(dsl_data.action_token_list[env]), i)
        exec(current_func_str)
        fn = eval('{}_r_action{}'.format(env.replace('-','_'), i+1))
        env_rules[env].append((token, fn))
        i += 1


def create_r_cste(number):
    def r_cste(t):
        return [('cste', lambda: number)]
    return r_cste


for i in range(20):
    rules.append(('R={}'.format(i), create_r_cste(i)))


def parse(program, environment='karel'):
    p_tokens = program.split()[::-1]
    queue = []
    applied = False
    while len(p_tokens) > 0 or len(queue) != 1:
        if applied: applied = False
        else:
            queue.append((p_tokens.pop(), None))
        for rule in rules + env_rules[environment]:
            applied = check_and_apply(queue, rule)
            if applied: break
        if not applied and len(p_tokens) == 0:  # error parsing
            return None, False
    return queue[0][1], True


