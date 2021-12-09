import numpy as np
from collections import defaultdict

import dsl_data

import types
import functools
def copy_func(f):
    """Based on http://stackoverflow.com/a/6528148/190597 (Glenn Maynard)"""
    g = types.FunctionType(f.__code__, f.__globals__, name=f.__name__,
            argdefs=f.__defaults__,
            closure=f.__closure__)
    g = functools.update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    return g

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


def r_prog_trace(t):
    stmt = t[3]

    def fn(k, n, record_dict, key):
        if n > MAX_FUNC_CALL: return k, n, False
        return stmt(k, n + 1, record_dict, stmt)
    return [('prog', fn)]
rules.append(('DEF run m( stmt m)', r_prog_trace))


def r_stmt_trace(t):
    stmt = t[0]

    def fn(k, n, record_dict, key):
        if n > MAX_FUNC_CALL: return k, n, False
        return stmt(k, n + 1, record_dict, stmt)
    return [('stmt', fn)]
rules.append(('while_stmt', r_stmt_trace))
rules.append(('repeat_stmt', r_stmt_trace))
rules.append(('stmt_stmt', r_stmt_trace))
rules.append(('action', r_stmt_trace))
rules.append(('if_stmt', r_stmt_trace))
rules.append(('ifelse_stmt', r_stmt_trace))


def r_stmt_stmt_trace(t):
    stmt1, stmt2 = t[0], t[1]

    def fn(k, n, record_dict, key):
        if n > MAX_FUNC_CALL: return k, n, False
        k, n, s = stmt1(k, n + 1, record_dict, stmt1)
        if not s: return k, n, s
        if n > MAX_FUNC_CALL: return k, n, False
        return stmt2(k, n, record_dict, stmt2)
    return [('stmt_stmt', fn)]
rules.append(('stmt stmt', r_stmt_stmt_trace))


def r_if_trace(t):
    cond, stmt = t[2], t[5]

    def fn(k, n, record_dict, key):
        if n > MAX_FUNC_CALL: return k, n, False
        k, n, s, c = cond(k, n + 1, record_dict, cond)
        assert len(record_dict[key]) == 1
        record_dict[key][0][1][c] = True
        if not s: return k, n, s
        if c: return stmt(k, n, record_dict, stmt)
        else: return k, n, s
    return [('if_stmt', copy_func(fn))]
rules.append(('IF c( cond c) i( stmt i)', r_if_trace))


def r_ifelse_trace(t):
    cond, stmt1, stmt2 = t[2], t[5], t[9]

    def fn(k, n, record_dict, key):
        if n > MAX_FUNC_CALL: return k, n, False
        k, n, s, c = cond(k, n + 1, record_dict, cond)
        #record_dict[key] = [(record_dict[key], c)]
        assert len(record_dict[key]) == 1
        record_dict[key][0][1][c] = True
        if not s: return k, n, s
        if c: return stmt1(k, n, record_dict, stmt1)
        else: return stmt2(k, n, record_dict, stmt2)
    return [('ifelse_stmt', copy_func(fn))]
rules.append(('IFELSE c( cond c) i( stmt i) ELSE e( stmt e)', r_ifelse_trace))


def r_while_trace(t):
    cond, stmt = t[2], t[5]

    def fn(k, n, record_dict, key):
        if n > MAX_FUNC_CALL: return k, n, False
        k, n, s, c = cond(k, n + 1, record_dict, cond)
        #record_dict[key] = [(record_dict[key], c)]
        assert len(record_dict[key]) == 1
        record_dict[key][0][1][c] = True
        if not s: return k, n, s
        while(c):
            k, n, s = stmt(k, n, record_dict, stmt)
            if not s: return k, n, s
            k, n, s, c = cond(k, n, record_dict, cond)
            assert len(record_dict[key]) == 1
            record_dict[key][0][1][c] = True
            if not s: return k, n, s
        return k, n, s
    return [('while_stmt', copy_func(fn))]
rules.append(('WHILE c( cond c) w( stmt w)', r_while_trace))


def r_repeat_trace(t):
    cste, stmt = t[1], t[3]

    def fn(k, n, record_dict, key):
        if n > MAX_FUNC_CALL: return k, n, False
        n += 1
        s = True
        for _ in range(cste()):
            k, n, s = stmt(k, n, record_dict, stmt)
            if not s: return k, n, s
        return k, n, s
    return [('repeat_stmt', fn)]
rules.append(('REPEAT cste r( stmt r)', r_repeat_trace))


def r_cond1_trace(t):
    cond = t[0]

    def fn(k, n, record_dict, key):
        if n > MAX_FUNC_CALL: return k, n, False, False
        return cond(k, n, record_dict, key)
    return [('cond', fn)]
rules.append(('cond_without_not', r_cond1_trace))


def r_cond2_trace(t):
    cond = t[2]

    def fn(k, n, record_dict, key):
        if n > MAX_FUNC_CALL: return k, n, False, False
        k, n, s, c = cond(k, n, record_dict, key)
        return k, n, s, not c
    return [('cond', fn)]
rules.append(('not c( cond c)', r_cond2_trace))


env_rules = defaultdict(list)
for env in dsl_data.envs:
    # Condition tokens
    func_str = '''
def {}_r_cond_without_not_{}_trace(t):
    def fn(k, n, record_dict, key):
        if n > MAX_FUNC_CALL: return k, n, False
        c = k.{}()
        return k, n, True, c
    return [('cond_without_not', fn)]
    '''

    for token, api in dsl_data.obv_token_api_dict[env].items():
        current_func_str = func_str.format(env.replace('-','_'), token, api)
        exec(current_func_str)
        fn = eval('{}_r_cond_without_not_{}_trace'.format(env.replace('-','_'), token))
        env_rules[env].append((token, fn))

    # Action tokens
    func_str = '''
def {}_r_action{}_trace(t):
    def fn(k, n, record_dict, key):
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
        fn = eval('{}_r_action{}_trace'.format(env.replace('-','_'), i+1))
        env_rules[env].append((token, fn))
        i += 1


def create_r_cste_trace(number):
    def r_cste_trace(t):
        return [('cste', lambda: number)]
    return r_cste_trace


for i in range(20):
    rules.append(('R={}'.format(i), create_r_cste_trace(i)))


def parse_and_trace(program, environment='karel'):
    record_dict = defaultdict(list)
    p_tokens = program.split()[::-1]
    p_current_lexpos = -1
    queue = []
    applied = False
    while len(p_tokens) > 0 or len(queue) != 1:
        if applied: applied = False
        else:
            queue.append((p_tokens.pop(), None))
            p_current_lexpos += 1
        for rule in rules + env_rules[environment]:
            applied = check_and_apply(queue, rule)
            if applied:
                if 'WHILE' in rule[0] or 'IF' in rule[0]:
                    #print("D:", rule, queue[-1][1], p_current_lexpos)
                    record_dict[queue[-1][1]].append([p_current_lexpos, {True: False, False: False}])
                break
        if not applied and len(p_tokens) == 0:  # error parsing
            return None, False, {}
    #import pprint
    #pprint.pprint(record_dict)
    return queue[0][1], True, record_dict


