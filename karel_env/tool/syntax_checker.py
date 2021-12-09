import torch
import numpy as np

STATE_MANDATORY_NEXT = 0
STATE_ACT_NEXT = 1
STATE_CSTE_NEXT = 2
STATE_BOOL_NEXT = 3
STATE_POSTCOND_OPEN_PAREN = 4


open_paren_token = ["m(", "c(", "r(", "w(", "i(", "e("]
close_paren_token = ["m)", "c)", "r)", "w)", "i)", "e)"]
flow_leads = ["REPEAT", "WHILE", "IF", "IFELSE"]
flow_need_bool = ["WHILE", "IF", "IFELSE"]
acts = ["move", "turnLeft", "turnRight", "pickMarker", "putMarker"]
bool_check = ["markersPresent", "noMarkersPresent", "leftIsClear", "rightIsClear", "frontIsClear"]
next_is_act = ["i(", "e(", "r(", "m(", "w("]
postcond_open_paren = ["i(", "w("]
possible_mandatories = ["DEF", "run", "c)", "ELSE", "<pad>"] + open_paren_token


def check_type(var, dtype):
    assert type(var) == dtype, 'data type should be {} but found {}'.format(dtype, type(var))


class CheckerState(object):

    def __init__(self, state, next_mandatory,
                 i_need_else_stack_pos, to_close_stack_pos,
                 c_deep, next_actblock_open):
        check_type(state, int)
        check_type(next_mandatory, int)
        check_type(i_need_else_stack_pos, int)
        check_type(to_close_stack_pos, int)
        check_type(c_deep, int)
        check_type(next_actblock_open, int)
        self.state = state
        self.next_mandatory = next_mandatory
        self.i_need_else_stack_pos = i_need_else_stack_pos
        self.to_close_stack_pos = to_close_stack_pos
        self.c_deep = c_deep
        self.next_actblock_open = next_actblock_open
        self.i_need_else_stack = torch.tensor(128 * [False], dtype=torch.bool)
        self.to_close_stack = 128 * [None]

    def __copy__(self):
        new_state = CheckerState(self.state, self.next_mandatory,
                                 self.i_need_else_stack_pos, self.to_close_stack_pos,
                                 self.c_deep, self.next_actblock_open)
        for i in range(0, self.i_need_else_stack_pos+1):
            new_state.i_need_else_stack[i] = self.i_need_else_stack[i]
        for i in range(0, self.to_close_stack_pos+1):
            new_state.to_close_stack[i] = self.to_close_stack[i]
        return new_state

    def push_closeparen_to_stack(self, close_paren):
        check_type(close_paren, int)
        self.to_close_stack_pos += 1
        self.to_close_stack[self.to_close_stack_pos] = close_paren

    def pop_close_paren(self):
        to_ret = self.to_close_stack[self.to_close_stack_pos]
        self.to_close_stack_pos -= 1
        check_type(to_ret, int)
        return to_ret

    def paren_to_close(self):
        return self.to_close_stack[self.to_close_stack_pos]

    def make_next_mandatory(self, next_mandatory):
        check_type(next_mandatory, int)
        self.state = STATE_MANDATORY_NEXT
        self.next_mandatory = next_mandatory

    def make_bool_next(self):
        self.state = STATE_BOOL_NEXT
        self.c_deep += 1

    def make_act_next(self):
        self.state = STATE_ACT_NEXT

    def close_cond_paren(self):
        self.c_deep -= 1
        if self.c_deep == 0:
            self.state = STATE_POSTCOND_OPEN_PAREN
        else:
            self.state = STATE_MANDATORY_NEXT
            # The mandatory next should already be "c)"

    def push_needelse_stack(self, need_else):
        check_type(need_else, bool)
        assert need_else == 0 or need_else == 1
        self.i_need_else_stack_pos += 1
        self.i_need_else_stack[self.i_need_else_stack_pos] = need_else

    def pop_needelse_stack(self):
        to_ret = self.i_need_else_stack[self.i_need_else_stack_pos]
        self.i_need_else_stack_pos -= 1
        # check_type(to_ret, torch.bool)
        return to_ret

    def set_next_actblock(self, next_actblock):
        check_type(next_actblock, int)
        self.next_actblock_open = next_actblock

    def make_next_cste(self):
        self.state = STATE_CSTE_NEXT


class SyntaxVocabulary(object):

    def __init__(self, def_tkn, run_tkn,
                 m_open_tkn, m_close_tkn,
                 else_tkn, e_open_tkn,
                 c_open_tkn, c_close_tkn,
                 i_open_tkn, i_close_tkn,
                 while_tkn, w_open_tkn,
                 repeat_tkn, r_open_tkn,
                 not_tkn, pad_tkn):
        self.def_tkn = def_tkn
        self.run_tkn = run_tkn
        self.m_open_tkn = m_open_tkn
        self.m_close_tkn = m_close_tkn
        self.else_tkn = else_tkn
        self.e_open_tkn = e_open_tkn
        self.c_open_tkn = c_open_tkn
        self.c_close_tkn = c_close_tkn
        self.i_open_tkn = i_open_tkn
        self.i_close_tkn = i_close_tkn
        self.while_tkn = while_tkn
        self.w_open_tkn = w_open_tkn
        self.repeat_tkn = repeat_tkn
        self.r_open_tkn = r_open_tkn
        self.not_tkn = not_tkn
        self.pad_tkn = pad_tkn


class PySyntaxChecker(object):

    def __init__(self, T2I, use_cuda, use_simplified_dsl=False, new_tokens=None):
        # check_type(args.no_cuda, bool)

        if use_simplified_dsl:
            global open_paren_token, close_paren_token, flow_leads, flow_need_bool, acts, bool_check
            global postcond_open_paren, possible_mandatories
            open_paren_token = [prl_tkn for prl_tkn in open_paren_token if prl_tkn in new_tokens]
            close_paren_token = [prl_tkn for prl_tkn in close_paren_token if prl_tkn in new_tokens]
            flow_leads = [prl_tkn for prl_tkn in flow_leads if prl_tkn in new_tokens]
            flow_need_bool = [prl_tkn for prl_tkn in flow_need_bool if prl_tkn in new_tokens]
            acts = [prl_tkn for prl_tkn in acts if prl_tkn in new_tokens]
            bool_check = [prl_tkn for prl_tkn in bool_check if prl_tkn in new_tokens]
            postcond_open_paren = [prl_tkn for prl_tkn in postcond_open_paren if prl_tkn in new_tokens]
            possible_mandatories = ["DEF", "run", "c)", "ELSE", "<pad>"] + open_paren_token
            possible_mandatories = [prl_tkn for prl_tkn in possible_mandatories if prl_tkn in new_tokens]
            # since we don't have DEF and run in simplified DSL, assign them a value that you will never see
            self.vocab = SyntaxVocabulary(len(T2I)+2, len(T2I)+2,
                                          T2I["m("], T2I["m)"], T2I["ELSE"], T2I["e("],
                                          T2I["c("], T2I["c)"], T2I["i("], T2I["i)"],
                                          T2I["WHILE"], T2I["w("], T2I["REPEAT"], T2I["r("],
                                          T2I["not"], T2I["<pad>"])
        else:
            self.vocab = SyntaxVocabulary(T2I["DEF"], T2I["run"],
                                          T2I["m("], T2I["m)"], T2I["ELSE"], T2I["e("],
                                          T2I["c("], T2I["c)"], T2I["i("], T2I["i)"],
                                          T2I["WHILE"], T2I["w("], T2I["REPEAT"], T2I["r("],
                                          T2I["not"], T2I["<pad>"])

        self.use_cuda = use_cuda
        self.open_parens = set([T2I[op] for op in open_paren_token])
        self.close_parens = set([T2I[op] for op in close_paren_token])
        self.if_statements = set([T2I[tkn] for tkn in ["IF", "IFELSE"]])
        self.op2cl = {}
        for op, cl in zip(open_paren_token, close_paren_token):
            self.op2cl[T2I[op]] = T2I[cl]
        self.need_else = {T2I["IF"]: False,
                          T2I["IFELSE"]: True}
        self.flow_lead = set([T2I[flow_lead_tkn] for flow_lead_tkn in flow_leads])
        self.effect_acts = set([T2I[act_tkn] for act_tkn in acts])
        self.act_acceptable = self.effect_acts | self.flow_lead | self.close_parens
        self.flow_needs_bool = set([T2I[flow_tkn] for flow_tkn in flow_need_bool])
        self.postcond_open_paren = set([T2I[op] for op in postcond_open_paren])
        self.range_cste = set([idx for tkn, idx in T2I.items() if tkn.startswith("R=")])
        self.bool_checks = set([T2I[bcheck] for bcheck in bool_check])

        tt = torch.cuda if use_cuda else torch
        self.vocab_size = len(T2I)
        self.mandatories_mask = {}
        for mand_tkn in possible_mandatories:
            mask = tt.BoolTensor(1,1,self.vocab_size).fill_(1)
            mask[0,0,T2I[mand_tkn]] = 0
            self.mandatories_mask[T2I[mand_tkn]] = mask
        self.act_next_masks = {}
        for close_tkn in self.close_parens:
            mask = tt.BoolTensor(1,1,self.vocab_size).fill_(1)
            mask[0,0,close_tkn] = 0
            for effect_idx in self.effect_acts:
                mask[0,0,effect_idx] = 0
            for flowlead_idx in self.flow_lead:
                mask[0,0,flowlead_idx] = 0
            self.act_next_masks[close_tkn] = mask
        self.range_mask = tt.BoolTensor(1,1,self.vocab_size).fill_(1)
        for ridx in self.range_cste:
            self.range_mask[0,0,ridx] = 0
        self.boolnext_mask = tt.BoolTensor(1,1,self.vocab_size).fill_(1)
        for bcheck_idx in self.bool_checks:
            self.boolnext_mask[0,0,bcheck_idx] = 0
        self.boolnext_mask[0,0,self.vocab.not_tkn] = 0
        self.postcond_open_paren_masks = {}
        for tkn in self.postcond_open_paren:
            mask = tt.BoolTensor(1,1,self.vocab_size).fill_(1)
            mask[0,0,tkn] = 0
            self.postcond_open_paren_masks[tkn] = mask

    def forward(self, state, new_idx):
        check_type(state, CheckerState)
        check_type(new_idx, int)
        # Whatever happens, if we open a paren, it needs to be closed
        if new_idx in self.open_parens:
            state.push_closeparen_to_stack(self.op2cl[new_idx])
        if new_idx in self.close_parens:
            paren_to_end = state.pop_close_paren()
            assert(new_idx == paren_to_end)

        if state.state == STATE_MANDATORY_NEXT:
            assert(new_idx == state.next_mandatory)
            if new_idx == self.vocab.def_tkn:
                state.make_next_mandatory(self.vocab.run_tkn)
            elif new_idx == self.vocab.run_tkn:
                state.make_next_mandatory(self.vocab.m_open_tkn)
            elif new_idx == self.vocab.else_tkn:
                state.make_next_mandatory(self.vocab.e_open_tkn)
            elif new_idx in self.open_parens:
                if new_idx == self.vocab.c_open_tkn:
                    state.make_bool_next()
                else:
                    state.make_act_next()
            elif new_idx == self.vocab.c_close_tkn:
                state.close_cond_paren()
            elif new_idx == self.vocab.pad_tkn:
                # Should this be at the top?
                # Keep the state in mandatory next, targetting <pad>
                # Once you go <pad>, you never go back.
                pass
            else:
                raise NotImplementedError

        elif state.state == STATE_ACT_NEXT:
            assert(new_idx in self.act_acceptable)

            if new_idx in self.flow_needs_bool:
                state.make_next_mandatory(self.vocab.c_open_tkn)
                # If we open one of the IF statements, we need to keep track if
                # it's one with a else statement or not
                if new_idx in self.if_statements:
                    state.push_needelse_stack(self.need_else[new_idx])
                    state.set_next_actblock(self.vocab.i_open_tkn)
                elif new_idx == self.vocab.while_tkn:
                    state.set_next_actblock(self.vocab.w_open_tkn)
                else:
                    raise NotImplementedError
            elif new_idx == self.vocab.repeat_tkn:
                state.make_next_cste()
            elif new_idx in self.effect_acts:
                pass
            elif new_idx in self.close_parens:
                if new_idx == self.vocab.i_close_tkn:
                    need_else = state.pop_needelse_stack()
                    if need_else:
                        state.make_next_mandatory(self.vocab.else_tkn)
                    else:
                        state.make_act_next()
                elif new_idx == self.vocab.m_close_tkn:
                    state.make_next_mandatory(self.vocab.pad_tkn)
                else:
                    state.make_act_next()
            else:
                raise NotImplementedError

        elif state.state == STATE_CSTE_NEXT:
            assert(new_idx in self.range_cste)
            state.make_next_mandatory(self.vocab.r_open_tkn)

        elif state.state == STATE_BOOL_NEXT:
            if new_idx in self.bool_checks:
                state.make_next_mandatory(self.vocab.c_close_tkn)
            elif new_idx == self.vocab.not_tkn:
                state.make_next_mandatory(self.vocab.c_open_tkn)
            else:
                raise NotImplementedError

        elif state.state == STATE_POSTCOND_OPEN_PAREN:
            assert(new_idx in self.postcond_open_paren)
            assert(new_idx == state.next_actblock_open)
            state.make_act_next()

        else:
            raise NotImplementedError

    def allowed_tokens(self, state):
        check_type(state, CheckerState)
        if state.state == STATE_MANDATORY_NEXT:
            # Only one possible token follows
            return self.mandatories_mask[state.next_mandatory]
        elif state.state == STATE_ACT_NEXT:
            # Either an action, a control flow statement or a closing of an open-paren
            return self.act_next_masks[state.paren_to_close()]
        elif state.state == STATE_CSTE_NEXT:
            return self.range_mask
        elif state.state == STATE_BOOL_NEXT:
            return self.boolnext_mask
        elif state.state == STATE_POSTCOND_OPEN_PAREN:
            return self.postcond_open_paren_masks[state.next_actblock_open]

    def get_sequence_mask(self, state, inp_sequence):
        check_type(state, CheckerState)
        check_type(inp_sequence, list)
        if len(inp_sequence) == 1:
            self.forward(state, inp_sequence[0])
            return self.allowed_tokens(state)
        else:
            tt = torch.cuda if self.use_cuda else torch
            mask_infeasible_list = []
            mask_infeasible = tt.BoolTensor(1, 1, self.vocab_size)
            for stp_idx, inp in enumerate(inp_sequence):
                self.forward(state, inp)
                mask_infeasible_list.append(self.allowed_tokens(state))
            torch.cat(mask_infeasible_list, 1, out=mask_infeasible)
            return mask_infeasible

    def get_initial_checker_state(self):
        return CheckerState(STATE_MANDATORY_NEXT, self.vocab.def_tkn,
                            -1, -1, 0, -1)

    def get_initial_checker_state2(self):
        return CheckerState(STATE_MANDATORY_NEXT, self.vocab.m_open_tkn,
                            -1, -1, 0, -1)


if __name__ == '__main__':
    import argparse
    import sys
    sys.path.insert(0, '.')
    from fetch_mapping import fetch_mapping

    parser = argparse.ArgumentParser(description='RL')
    # remap prl tokens to dsl tokens
    parser.add_argument('--mapping_file',
                        default='mapping_karel2prl.txt',
                        type=str)
    args = parser.parse_args()

    # fetch the mapping from prl tokens to dsl tokens
    if args.mapping_file is not None:
        args.dsl2prl_mapping, args.prl2dsl_mapping, args.dsl_tokens, args.prl_tokens = \
            fetch_mapping(args.mapping_file)
        args.use_simplified_dsl = True
        args.use_shorter_if = True if 'shorter_if' in args.mapping_file else False
    else:
        args.use_simplified_dsl = False

    T2I = {token: i for i, token in enumerate(args.dsl_tokens)}
    I2T = {i: token for i, token in enumerate(args.dsl_tokens)}
    T2I['<pad>'] = len(args.dsl_tokens)
    I2T[len(args.dsl_tokens)] = '<pad>'
    sample_program = [0, 1, 2, 49, 32, 41, 33, 47, 31, 13, 9, 8, 10, 4, 4, 48, 3, len(args.dsl_tokens)]
    sample_program = [0, 1, 2, 38, ]
    use_cuda = False

    syntax_checker = PySyntaxChecker(T2I, use_cuda)
    initial_state = syntax_checker.get_initial_checker_state()
    sequence_mask = syntax_checker.get_sequence_mask(initial_state, sample_program).squeeze()
    for idx, token in enumerate(sample_program):
        valid_tokens = torch.where(sequence_mask[idx] == 0)[0]
        valid_tokens = [I2T[tkn.detach().cpu().numpy().tolist()] for tkn in valid_tokens]
        valid_tokens = " ".join(valid_tokens)
        print("valid tokens for {}: {}".format(I2T[token], valid_tokens))




