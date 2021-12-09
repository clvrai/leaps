import sys
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import gym

from karel_env.tool.syntax_checker import PySyntaxChecker
from karel_env.karel_supervised import Karel_world_supervised

from rl.distributions import FixedCategorical, FixedNormal
from rl.model import NNBase
from rl.utils import masked_mean, masked_sum, create_hook, init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

#TODO: replace _unmask_idx with _unmask_idx2 after verifying identity
def _unmask_idx(output_mask_all, first_end_token_idx, max_program_len):
    for p_idx in range(first_end_token_idx.shape[0]):
        t_idx = int(first_end_token_idx[p_idx].detach().cpu().numpy())
        if t_idx < max_program_len:
            output_mask_all[p_idx, t_idx] = True
    return output_mask_all.to(torch.bool)

def _unmask_idx2(x):
    seq, seq_len = x
    if seq_len < seq.shape[0]:
        seq[seq_len] = True
        return True
    return False


class Normal(nn.Module):
    def __init__(self):
        super(Normal, self).__init__()

    def forward(self, mean, std):
        return FixedNormal(mean, std)


class Encoder(NNBase):
    def __init__(self, num_inputs, num_outputs, recurrent=True, hidden_size=64, rnn_type='GRU', two_head=False):
        super(Encoder, self).__init__(recurrent, num_inputs, hidden_size, rnn_type)

        self._rnn_type = rnn_type
        self._two_head = two_head
        self.token_encoder = nn.Embedding(num_inputs, num_inputs)

    def forward(self, src, src_len):
        program_embeddings = self.token_encoder(src)
        src_len = src_len.cpu()
        packed_embedded = nn.utils.rnn.pack_padded_sequence(program_embeddings, src_len, batch_first=True,
                                                            enforce_sorted=False)

        if self.is_recurrent:
            x, rnn_hxs = self.gru(packed_embedded)

        return x, rnn_hxs

class Decoder(NNBase):
    def __init__(self, num_inputs, num_outputs, recurrent=False, hidden_size=64, rnn_type='GRU', two_head=False, **kwargs):
        super(Decoder, self).__init__(recurrent, num_inputs+hidden_size, hidden_size, rnn_type)

        self._rnn_type = rnn_type
        self._two_head = two_head
        self.num_inputs = num_inputs
        self.use_simplified_dsl = kwargs['dsl']['use_simplified_dsl']
        self.max_program_len = kwargs['dsl']['max_program_len']
        self.grammar = kwargs['grammar']
        self.num_program_tokens = kwargs['num_program_tokens']
        self.setup = kwargs['algorithm']
        self.rl_algorithm = kwargs['rl']['algo']['name']
        self.value_method = kwargs['rl']['value_method']
        self.value_embedding = 'eop_rnn'

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.token_encoder = nn.Embedding(num_inputs, num_inputs)

        self.token_output_layer = nn.Sequential(
            init_(nn.Linear(hidden_size + num_inputs + hidden_size, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, num_outputs)))

        # This check is required only to support backward compatibility to pre-trained models
        if (self.setup =='RL' or self.setup =='supervisedRL') and kwargs['rl']['algo']['name'] != 'reinforce':
            self.critic = nn.Sequential(
                init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
                init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

            self.critic_linear = init_(nn.Linear(hidden_size, 1))

        if self._two_head:
            self.eop_output_layer = nn.Sequential(
                init_(nn.Linear(hidden_size + num_inputs + hidden_size, hidden_size)), nn.Tanh(),
                init_(nn.Linear(hidden_size, 2)))

        self._init_syntax_checker(kwargs)

        self.softmax = nn.LogSoftmax(dim=-1)

        self.train()

    def _init_syntax_checker(self, config):
        # use syntax checker to check grammar of output program prefix
        if self.use_simplified_dsl:
            self.prl_tokens = config['prl_tokens']
            self.dsl_tokens = config['dsl_tokens']
            self.prl2dsl_mapping = config['prl2dsl_mapping']
            syntax_checker_tokens = copy.copy(config['prl_tokens'])
        else:
            syntax_checker_tokens = copy.copy(config['dsl_tokens'])

        T2I = {token: i for i, token in enumerate(syntax_checker_tokens)}
        T2I['<pad>'] = len(syntax_checker_tokens)
        self.T2I = T2I
        syntax_checker_tokens.append('<pad>')
        if self.grammar == 'handwritten':
            self.syntax_checker = PySyntaxChecker(T2I, use_cuda='cuda' in config['device'],
                                                  use_simplified_dsl=self.use_simplified_dsl,
                                                  new_tokens=syntax_checker_tokens)

    def _forward_one_pass(self, current_tokens, context, rnn_hxs, masks):
        token_embedding = self.token_encoder(current_tokens)
        inputs = torch.cat((token_embedding, context), dim=-1)

        if self.is_recurrent:
            outputs, rnn_hxs = self._forward_rnn(inputs, rnn_hxs, masks.view(-1, 1))

        pre_output = torch.cat([outputs, token_embedding, context], dim=1)
        output_logits = self.token_output_layer(pre_output)

        value = None
        if (self.setup =='RL' or self.setup =='supervisedRL') and self.rl_algorithm != 'reinforce':
            hidden_critic = self.critic(rnn_hxs)
            value = self.critic_linear(hidden_critic)

        eop_output_logits = None
        if self._two_head:
            eop_output_logits = self.eop_output_layer(pre_output)
        return value, output_logits, rnn_hxs, eop_output_logits

    def _temp_init(self, batch_size, device):
        # create input with token as DEF
        inputs = torch.ones((batch_size)).to(torch.long).to(device)
        inputs = (0 * inputs)# if self.use_simplified_dsl else (2 * inputs)

        # input to the GRU
        gru_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        return inputs, gru_mask

    def _get_syntax_mask(self, batch_size, current_tokens, mask_size, grammar_state):
        out_of_syntax_list = []
        device = current_tokens.device
        out_of_syntax_mask = torch.zeros((batch_size, mask_size),
                                         dtype=torch.bool, device=device)

        for program_idx, inp_token in enumerate(current_tokens):
            inp_dsl_token = inp_token.detach().cpu().numpy().item()
            out_of_syntax_list.append(self.syntax_checker.get_sequence_mask(grammar_state[program_idx],
                                                                            [inp_dsl_token]).to(device))
        torch.cat(out_of_syntax_list, 0, out=out_of_syntax_mask)
        out_of_syntax_mask = out_of_syntax_mask.squeeze()
        syntax_mask = torch.where(out_of_syntax_mask,
                                  -torch.finfo(torch.float32).max * torch.ones_like(out_of_syntax_mask).float(),
                                  torch.zeros_like(out_of_syntax_mask).float())

        # If m) is not part of next valid tokens in syntax_mask then only eop action can be eop=0 otherwise not
        # use absence of m) to mask out eop = 1, use presence of m) and eop=1 to mask out all tokens except m)
        eop_syntax_mask = None
        if self._two_head:
            # use absence of m) to mask out eop = 1
            gather_m_closed = torch.tensor(batch_size * [self.T2I['m)']], dtype=torch.long, device=device).view(-1, 1)
            eop_in_valid_set = torch.gather(syntax_mask, 1, gather_m_closed)
            eop_syntax_mask = torch.zeros((batch_size, 2), device=device)
            # if m) is absent we can't predict eop=1
            eop_syntax_mask[:, 1] = eop_in_valid_set.flatten()

        return syntax_mask, eop_syntax_mask, grammar_state

    def _get_eop_preds(self, eop_output_logits, eop_syntax_mask, syntax_mask, output_mask, deterministic=False):
        batch_size = eop_output_logits.shape[0]
        device = eop_output_logits.device

        # eop_action
        if eop_syntax_mask is not None:
            assert eop_output_logits.shape == eop_syntax_mask.shape
            eop_output_logits += eop_syntax_mask
        if self.setup == 'supervised':
            eop_preds = self.softmax(eop_output_logits).argmax(dim=-1).to(torch.bool)
        elif self.setup == 'RL':
            # define distribution over current logits
            eop_dist = FixedCategorical(logits=eop_output_logits)
            # sample actions
            eop_preds = eop_dist.mode() if deterministic else eop_dist.sample()
        else:
            raise NotImplementedError()


        #  use presence of m) and eop=1 to mask out all tokens except m)
        if self.grammar != 'None':
            new_output_mask = (~(eop_preds.to(torch.bool))) * output_mask
            assert output_mask.dtype == torch.bool
            output_mask_change = (new_output_mask != output_mask).view(-1, 1)
            output_mask_change_repeat = output_mask_change.repeat(1, syntax_mask.shape[1])
            new_syntax_mask = -torch.finfo(torch.float32).max * torch.ones_like(syntax_mask).float()
            new_syntax_mask[:, self.T2I['m)']] = 0
            syntax_mask = torch.where(output_mask_change_repeat, new_syntax_mask, syntax_mask)

        return eop_preds, eop_output_logits, syntax_mask

    def forward(self, gt_programs, embeddings, teacher_enforcing=True, action=None, output_mask_all=None,
                eop_action=None, deterministic=False, evaluate=False, max_program_len=float('inf')):
        if self.setup == 'supervised':
            assert deterministic == True
        batch_size, device = embeddings.shape[0], embeddings.device
        # NOTE: for pythorch >=1.2.0, ~ only works correctly on torch.bool
        if evaluate:
            output_mask = output_mask_all[:, 0]
        else:
            output_mask = torch.ones(batch_size).to(torch.bool).to(device)

        current_tokens, gru_mask = self._temp_init(batch_size, device)
        if self._rnn_type == 'GRU':
            rnn_hxs = embeddings
        elif self._rnn_type == 'LSTM':
            rnn_hxs = (embeddings, embeddings)
        else:
            raise NotImplementedError()

        # Encode programs
        max_program_len = min(max_program_len, self.max_program_len)
        value_all = []
        pred_programs = []
        pred_programs_log_probs_all = []
        dist_entropy_all = []
        eop_dist_entropy_all = []
        output_logits_all = []
        eop_output_logits_all = []
        eop_pred_programs = []
        if not evaluate:
            output_mask_all = torch.ones(batch_size, self.max_program_len).to(torch.bool).to(device)
        first_end_token_idx = self.max_program_len * torch.ones(batch_size).to(device)

        # using get_initial_checker_state2 because we skip prediction for 'DEF', 'run' tokens
        if self.grammar == 'handwritten':
            if self.use_simplified_dsl:
                grammar_state = [self.syntax_checker.get_initial_checker_state2()
                                 for _ in range(batch_size)]
            else:
                grammar_state = [self.syntax_checker.get_initial_checker_state()
                                 for _ in range(batch_size)]

        for i in range(max_program_len):
            value, output_logits, rnn_hxs, eop_output_logits = self._forward_one_pass(current_tokens, embeddings,
                                                                                      rnn_hxs, gru_mask)

            # limit possible actions using syntax checker if available
            # action_logits * syntax_mask where syntax_mask = {-inf, 0}^|num_program_tokens|
            # syntax_mask = 0  for action a iff for given input(e.g.'DEF'), a(e.g.'run') creates a valid program prefix
            syntax_mask = None
            eop_syntax_mask = None
            if self.grammar != 'None':
                mask_size = output_logits.shape[1]
                syntax_mask, eop_syntax_mask, grammar_state = self._get_syntax_mask(batch_size, current_tokens,
                                                                                    mask_size, grammar_state)

            # get eop action and new syntax mask if using syntax checker
            if self._two_head:
                eop_preds, eop_output_logits, syntax_mask = self._get_eop_preds(eop_output_logits, eop_syntax_mask,
                                                                                syntax_mask, output_mask_all[:, i])

            # apply softmax
            if syntax_mask is not None:
                assert (output_logits.shape == syntax_mask.shape) or self.setup == 'CEM', '{}:{}'.format(output_logits.shape, syntax_mask.shape)
                output_logits += syntax_mask
            if self.setup == 'supervised' or self.setup == 'CEM':
                preds = self.softmax(output_logits).argmax(dim=-1)
            elif self.setup == 'RL':
                # define distribution over current logits
                dist = FixedCategorical(logits=output_logits)
                # sample actions
                preds = dist.mode().squeeze() if deterministic else dist.sample().squeeze()
                # calculate log probabilities
                if evaluate:
                    assert action[:,i].shape == preds.shape
                    pred_programs_log_probs = dist.log_probs(action[:,i])
                else:
                    pred_programs_log_probs = dist.log_probs(preds)

                if self._two_head:
                    raise NotImplementedError()
                # calculate entropy
                dist_entropy = dist.entropy()
                if self._two_head:
                    raise NotImplementedError()
                pred_programs_log_probs_all.append(pred_programs_log_probs)
                dist_entropy_all.append(dist_entropy.view(-1, 1))
            else:
                raise NotImplementedError()

            # calculate mask for current tokens
            assert preds.shape == output_mask.shape
            if not evaluate:
                if self._two_head:
                    output_mask = (~(eop_preds.to(torch.bool))) * output_mask
                else:
                    output_mask = (~((preds == self.num_program_tokens - 1).to(torch.bool))) * output_mask

                # recalculate first occurrence of <pad> for each program
                first_end_token_idx = torch.min(first_end_token_idx,
                                                ((self.max_program_len * output_mask.float()) +
                                                 ((1 - output_mask.float()) * i)).flatten())

            value_all.append(value)
            output_logits_all.append(output_logits)
            pred_programs.append(preds)
            if self._two_head:
                eop_output_logits_all.append(eop_output_logits)
                eop_pred_programs.append(eop_preds)
            if not evaluate:
                output_mask_all[:, i] = output_mask.flatten()

            if self.setup == 'supervised':
                if teacher_enforcing:
                    current_tokens = gt_programs[:, i+1].squeeze()
                else:
                    current_tokens = preds.squeeze()
            else:
                if evaluate:
                    assert self.setup == 'RL'
                    current_tokens = action[:, i]
                else:
                    current_tokens = preds.squeeze()


        # umask first end-token for two headed policy
        if not evaluate:
            output_mask_all = _unmask_idx(output_mask_all, first_end_token_idx, self.max_program_len).detach()

        # combine all token parameters to get program parameters
        raw_pred_programs_all = torch.stack(pred_programs, dim=1)
        raw_output_logits_all = torch.stack(output_logits_all, dim=1)
        pred_programs_len = torch.sum(output_mask_all, dim=1, keepdim=True)

        if not self._two_head:
            assert output_mask_all.dtype == torch.bool
            pred_programs_all = torch.where(output_mask_all, raw_pred_programs_all,
                                            int(self.num_program_tokens - 1) * torch.ones_like(raw_pred_programs_all))
            eop_pred_programs_all = -1 * torch.ones_like(pred_programs_all)
            raw_eop_output_logits_all = None
        else:
            pred_programs_all = raw_pred_programs_all
            eop_pred_programs_all = torch.stack(eop_pred_programs, dim=1)
            raw_eop_output_logits_all = torch.stack(eop_output_logits_all, dim=1)

        # calculate log_probs, value, actions for program from token values
        if self.setup == 'RL':
            raw_pred_programs_log_probs_all = torch.cat(pred_programs_log_probs_all, dim=1)
            pred_programs_log_probs_all = masked_sum(raw_pred_programs_log_probs_all, output_mask_all,
                                                     dim=1, keepdim=True)

            raw_dist_entropy_all = torch.cat(dist_entropy_all, dim=1)
            dist_entropy_all = masked_mean(raw_dist_entropy_all, output_mask_all,
                                           dim=tuple(range(len(output_mask_all.shape))))

            # calculate value for program from token values
            if self.rl_algorithm != 'reinforce':
                if self.value_method == 'mean':
                    raw_value_all = torch.cat(value_all, dim=1)
                    value_all = masked_mean(raw_value_all, output_mask_all, dim=1, keepdim=True)
                else:
                    # calculate value function from hidden states
                    raw_value_all = torch.cat(value_all, dim=1)
                    value_idx = torch.sum(output_mask_all, dim=1, keepdim=True) - 1
                    assert len(value_idx.shape) == 2 and value_idx.shape[1] == 1
                    value_all = torch.gather(raw_value_all, 1, value_idx)

                    # This value calculation is just for sanity check
                    with torch.no_grad():
                        value_idx_2 = first_end_token_idx.clamp(max=self.max_program_len - 1).long().reshape(-1, 1)
                        value_all_2 = torch.gather(raw_value_all, 1, value_idx_2)
                        assert torch.sum(value_all != value_all_2) == 0
                    assert value_all.shape[0] == batch_size
            else:
                value_all = torch.zeros_like(pred_programs_log_probs_all)
        else:
            dist_entropy_all = None
            value_all = None

        return value_all, pred_programs_all, pred_programs_len, pred_programs_log_probs_all, raw_output_logits_all,\
               eop_pred_programs_all, raw_eop_output_logits_all, output_mask_all, dist_entropy_all


class VAE(torch.nn.Module):

    def __init__(self, num_inputs, num_program_tokens, **kwargs):
        super(VAE, self).__init__()
        self._two_head = kwargs['two_head']
        self._vanilla_ae = kwargs['AE']
        self._tanh_after_mu_sigma = kwargs['net']['tanh_after_mu_sigma']
        self._tanh_after_sample = kwargs['net']['tanh_after_sample']
        self._use_latent_dist = not kwargs['net']['controller']['use_decoder_dist']
        self._rnn_type = kwargs['net']['rnn_type']
        num_outputs = num_inputs

        self.encoder = Encoder(num_inputs, num_outputs, recurrent=kwargs['recurrent_policy'],
                               hidden_size=kwargs['num_lstm_cell_units'], rnn_type=kwargs['net']['rnn_type'],
                               two_head=kwargs['two_head'])
        self.decoder = Decoder(num_inputs, num_outputs, recurrent=kwargs['recurrent_policy'],
                               hidden_size=kwargs['num_lstm_cell_units'], rnn_type=kwargs['net']['rnn_type'],
                               num_program_tokens=num_program_tokens, **kwargs)
        self._enc_mu = torch.nn.Linear(kwargs['num_lstm_cell_units'], kwargs['num_lstm_cell_units'])
        self._enc_log_sigma = torch.nn.Linear(kwargs['num_lstm_cell_units'], kwargs['num_lstm_cell_units'])
        self.tanh = torch.nn.Tanh()

    @property
    def latent_dim(self):
        return  self._enc_mu.out_features

    def _sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = self._enc_mu(h_enc)
        log_sigma = self._enc_log_sigma(h_enc)
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).to(torch.float).to(h_enc.device)
        if self._tanh_after_mu_sigma: #False by default
            mu = self.tanh(mu)
            sigma = self.tanh(sigma)

        self.z_mean = mu
        self.z_sigma = sigma

        return mu + sigma * Variable(std_z, requires_grad=False)  # Reparameterization trick

    @staticmethod
    def latent_loss(z_mean, z_stddev):
        mean_sq = z_mean * z_mean
        stddev_sq = z_stddev * z_stddev
        return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)

    def forward(self, programs, program_masks, teacher_enforcing, deterministic=True):
        program_lens = program_masks.squeeze().sum(dim=-1)
        _, h_enc = self.encoder(programs, program_lens)

        if self._rnn_type == 'GRU':
            z = h_enc.squeeze() if self._vanilla_ae else self._sample_latent(h_enc.squeeze())
        elif self._rnn_type == 'LSTM':
            z = h_enc[0].squeeze() if self._vanilla_ae else self._sample_latent(h_enc[0].squeeze())
        else:
            raise NotImplementedError()

        if self._tanh_after_sample:
            z = self.tanh(z)
        return self.decoder(programs, z, teacher_enforcing=teacher_enforcing, deterministic=deterministic), z


class ConditionPolicy(NNBase):
    def __init__(self, envs, **kwargs):
        hidden_size = kwargs['num_lstm_cell_units']
        rnn_type = kwargs['net']['rnn_type']
        recurrent = kwargs['recurrent_policy']
        self.num_agent_actions = kwargs['dsl']['num_agent_actions']
        super(ConditionPolicy, self).__init__(recurrent, 2 * hidden_size + self.num_agent_actions, hidden_size, rnn_type)

        self.envs = envs
        self.state_shape = (16, kwargs['height'], kwargs['width'])
        self._hidden_size = hidden_size
        self._recurrent = recurrent
        self._rnn_type = rnn_type
        self.max_demo_length = kwargs['max_demo_length']
        self.setup = kwargs['algorithm']
        self.rl_algorithm = kwargs['rl']['algo']['name']
        self.value_method = kwargs['rl']['value_method']
        self.value_embedding = 'eop_rnn'
        self.use_teacher_enforcing =  kwargs['net']['condition']['use_teacher_enforcing']
        self.states_source = kwargs['net']['condition']['observations']

        self._world = Karel_world_supervised(s=None, make_error=False)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.action_encoder = nn.Embedding(self.num_agent_actions, self.num_agent_actions)

        self.state_encoder = nn.Sequential(
            init_(nn.Conv2d(self.state_shape[0], 32, 3, stride=1)), nn.ReLU(),
            init_(nn.Conv2d(32, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * 4 * 4, hidden_size)), nn.ReLU())

        self.mlp = nn.Sequential(init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
                                 init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
                                 init_(nn.Linear(hidden_size, self.num_agent_actions)))

        # This check is required only to support backward compatibility to pre-trained models
        if (self.setup =='RL' or self.setup =='supervisedRL') and kwargs['rl']['algo']['name'] != 'reinforce':
            self.critic = nn.Sequential(
                init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
                init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

            self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.softmax = nn.LogSoftmax(dim=-1)

        self.train()


    def _forward_one_pass(self, inputs, rnn_hxs, masks):
        if self.is_recurrent:
            mlp_inputs, rnn_hxs = self._forward_rnn(inputs, rnn_hxs, masks)

        logits = self.mlp(mlp_inputs)

        value = None
        if (self.setup =='RL' or self.setup =='supervisedRL') and self.rl_algorithm != 'reinforce':
            hidden_critic = self.critic(rnn_hxs)
            value = self.critic_linear(hidden_critic)

        return value, logits, rnn_hxs

    def _env_step(self, states, actions, step):
        states = states.detach().cpu().numpy().astype(np.bool_)
        # C x H x W to H x W x C
        states = np.moveaxis(states,[-1,-2,-3], [-2,-3,-1])
        assert states.shape[-1] == 16
        # karel world expects H x W x C
        if step == 0:
            self._world.reset(states)
        new_states = self._world.step(actions.detach().cpu().numpy())
        new_states = np.moveaxis(new_states,[-1,-2,-3], [-3,-1,-2])
        new_states = torch.tensor(new_states, dtype=torch.float32, device=actions.device)
        return new_states


    def forward(self, s_h, a_h, z, teacher_enforcing=True, eval_actions=None, eval_masks_all=None,
                deterministic=False, evaluate=False):
        """

        :param s_h:
        :param a_h:
        :param z:
        :param teacher_enforcing: True if training in supervised setup or evaluating actions in RL setup
        :param eval_actions:
        :param eval_masks_all:
        :param deterministic:
        :param evaluate: True if setup == RL and evaluating actions, False otherwise
        :return:
        """
        if self.setup == 'supervised':
            assert deterministic == True
        # s_h: B x num_demos_per_program x 1 x C x H x W
        batch_size, num_demos_per_program, demo_len, C, H, W = s_h.shape
        new_batch_size = s_h.shape[0] * s_h.shape[1]
        teacher_enforcing = teacher_enforcing and self.use_teacher_enforcing
        old_states = s_h.squeeze().view(new_batch_size, C, H, W)

        """ get state_embedding of one image per demonstration"""
        state_embeddings = self.state_encoder(s_h[:, :, 0, :, :, :].view(new_batch_size, C, H, W))
        state_embeddings = state_embeddings.view(batch_size, num_demos_per_program, self._hidden_size)
        assert state_embeddings.shape[0] == batch_size and state_embeddings.shape[1] == num_demos_per_program
        state_embeddings = state_embeddings.squeeze()

        """ get intention_embeddings"""
        intention_embedding = z.unsqueeze(1).repeat(1, num_demos_per_program, 1)

        """ get action embeddings for initial actions"""
        actions = (self.num_agent_actions - 1) * torch.ones((batch_size * num_demos_per_program, 1), device=s_h.device,
                                                            dtype=torch.long)

        rnn_hxs = intention_embedding.view(batch_size * num_demos_per_program, self._hidden_size)
        masks = torch.ones((batch_size * num_demos_per_program, 1), device=intention_embedding.device, dtype=torch.bool)
        gru_mask = torch.ones((batch_size * num_demos_per_program, 1), device=intention_embedding.device, dtype=torch.bool)
        assert rnn_hxs.shape[0] == gru_mask.shape[0]
        if self._rnn_type == 'LSTM':
            rnn_hxs = (rnn_hxs, rnn_hxs)
        masks_all = []
        value_all = []
        actions_all = []
        action_logits_all = []
        action_log_probs_all = []
        dist_entropy_all = []
        max_a_h_len = self.max_demo_length-1
        for i in range(self.max_demo_length-1):
            """ get action embeddings and concatenate them with intention and state embeddings """
            action_embeddings = self.action_encoder(actions.view(batch_size, num_demos_per_program))
            inputs = torch.cat((intention_embedding, state_embeddings, action_embeddings), dim=-1)
            inputs = inputs.view(batch_size * num_demos_per_program, -1)

            """ forward pass"""
            value, action_logits, rnn_hxs = self._forward_one_pass(inputs, rnn_hxs, gru_mask)

            """ apply a temporary softmax to get action values to calculate masks """
            if self.setup == 'supervised':
                with torch.no_grad():
                    actions = self.softmax(action_logits).argmax(dim=-1).view(-1, 1)
            elif self.setup == 'RL':
                # define distribution over current logits
                dist = FixedCategorical(logits=action_logits)
                # calculate log probabilities
                if evaluate:
                    assert eval_actions[:, i].shape == actions.squeeze().shape, '{}:{}'.format(eval_actions[:, i].shape,
                                                                                               actions.squeeze().shape)
                    action_log_probs = dist.log_probs(eval_actions[:,i])
                else:
                    # sample actions
                    actions = dist.mode() if deterministic else dist.sample()
                    action_log_probs = dist.log_probs(actions)

                # calculate entropy
                dist_entropy = dist.entropy()
                action_log_probs_all.append(action_log_probs)
                dist_entropy_all.append(dist_entropy.view(-1,1))
            else:
                raise NotImplementedError()

            assert masks.shape == actions.shape
            if not evaluate:
                # NOTE: remove this if check and keep mask update line in case we want to speed up training
                if masks.detach().sum().cpu().item() != 0:
                    masks = masks  * (actions < 5)
                masks_all.append(masks)

            value_all.append(value)
            action_logits_all.append(action_logits)
            actions_all.append(actions)

            """ apply teacher enforcing if ground-truth trajectories available """
            if teacher_enforcing:
                if self.setup == 'supervised':
                    actions = a_h[:, :, i].squeeze().long().view(-1, 1)
                else:
                    actions = eval_actions[:, i].squeeze().long().view(-1, 1)

            """ get the next state embeddings for input to the network"""
            if self.states_source != 'initial_state':
                if teacher_enforcing and self.states_source == 'dataset':
                    new_states = s_h[:, :, i+1, :, :, :].view(s_h.shape[0] * s_h.shape[1], C, H, W)
                else:
                    new_states = self._env_step(old_states, actions, i)
                    assert new_states.shape == (batch_size * num_demos_per_program, C, H, W)

                state_embeddings = self.state_encoder(new_states).view(batch_size, num_demos_per_program,
                                                                         self._hidden_size)
                old_states = new_states

        # unmask first <pad> token
        if not evaluate:
            masks_all = torch.stack(masks_all, dim=1).squeeze()
            first_end_token_idx = torch.sum(masks_all.squeeze(), dim=1)
            _ = list(map(_unmask_idx2, zip(masks_all, first_end_token_idx)))

        action_logits_all = torch.stack(action_logits_all, dim=1)
        assert action_logits_all.shape[-1] == 6

        if self.setup == 'RL':
            masks_all = eval_masks_all if evaluate else masks_all
            actions_all = torch.cat(actions_all, dim=1)

            raw_action_log_probs_all = torch.cat(action_log_probs_all, dim=1)
            action_log_probs_all = masked_sum(raw_action_log_probs_all, masks_all, dim=1, keepdim=True)

            raw_dist_entropy_all = torch.cat(dist_entropy_all, dim=1)
            dist_entropy_all = masked_mean(raw_dist_entropy_all, masks_all, dim=tuple(range(len(masks_all.shape))))

            # calculate value for program from token values
            if self.rl_algorithm != 'reinforce':
                if self.value_method == 'mean':
                    raw_value_all = torch.cat(value_all, dim=1)
                    value_all = masked_mean(raw_value_all, masks_all, dim=1, keepdim=True)
                else:
                    # calculate value function from hidden states
                    raw_value_all = torch.cat(value_all, dim=1)
                    value_idx = torch.sum(masks_all, dim=1, keepdim=True) - 1
                    assert len(value_idx.shape) == 2 and value_idx.shape[1] == 1
                    value_all = torch.gather(raw_value_all, 1, value_idx)

                    # this value calculation is just for sanity check
                    with torch.no_grad():
                        value_idx_2 = first_end_token_idx.clamp(max=self.max_program_len - 1).long().reshape(-1, 1)
                        value_all_2 = torch.gather(raw_value_all, 1, value_idx_2)
                        assert torch.sum(value_all != value_all_2) == 0
                    assert value_all.shape[0] == batch_size
            else:
                value_all = torch.zeros_like(action_log_probs_all)

            value_all = value_all.view(batch_size, num_demos_per_program, 1)
            actions_all = actions_all.view(batch_size, num_demos_per_program, self.max_demo_length - 1)
            masks_all = masks_all.view(batch_size, num_demos_per_program, self.max_demo_length - 1)
            action_log_probs_all = action_log_probs_all.view(batch_size, num_demos_per_program, 1)

        else:
            value_all = None

        return value_all, actions_all, action_log_probs_all, action_logits_all, masks_all, dist_entropy_all


class ProgramVAE(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ProgramVAE, self).__init__()
        envs = args[0]
        action_space = envs.action_space
        num_outputs = int(action_space.high[0]) if not kwargs['two_head'] else int(action_space.high[0] - 1)
        num_program_tokens = num_outputs if not kwargs['two_head'] else num_outputs + 1
        # two_head policy shouldn't have <pad> token in action distribution, but syntax checker forces it
        # even if its included, <pad> will always have masked probability = 0, so implementation vise it should be fine
        if kwargs['two_head'] and kwargs['grammar'] == 'handwritten':
            num_outputs = int(action_space.high[0])

        self._tanh_after_sample = kwargs['net']['tanh_after_sample']
        self._debug = kwargs['debug']
        self.use_decoder_dist = kwargs['net']['controller']['use_decoder_dist']
        self.use_condition_policy_in_rl = kwargs['rl']['loss']['condition_rl_loss_coef'] > 0.0
        self.num_demo_per_program = kwargs['rl']['envs']['executable']['num_demo_per_program']
        self.max_demo_length = kwargs['rl']['envs']['executable']['max_demo_length']

        self.num_program_tokens = num_program_tokens
        self.teacher_enforcing = kwargs['net']['decoder']['use_teacher_enforcing']
        self.vae = VAE(num_outputs, num_program_tokens, **kwargs)
        self.condition_policy = ConditionPolicy(envs, **kwargs)

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.vae.latent_dim

    @property
    def is_recurrent(self):
        return self.vae.encoder.is_recurrent

    def forward(self, programs, program_masks, init_states, a_h, rnn_hxs=None, masks=None, action=None, output_mask_all=None, eop_action=None,
                agent_actions=None, agent_action_masks=None, deterministic=False, evaluate=False):

        if self.vae.decoder.setup == 'supervised':
            output, z = self.vae(programs, program_masks, self.teacher_enforcing, deterministic=deterministic)
            _, pred_programs, pred_programs_len, _, output_logits, eop_pred_programs, eop_output_logits, pred_program_masks, _ = output
            _, _, _, action_logits, action_masks, _ = self.condition_policy(init_states, a_h, z, self.teacher_enforcing,
                                                                         deterministic=deterministic)
            return pred_programs, pred_programs_len, output_logits, eop_pred_programs, eop_output_logits, \
                   pred_program_masks, action_logits, action_masks, z

        # output, z = self.vae(programs, program_masks, self.teacher_enforcing)
        """ VAE forward pass """
        program_lens = program_masks.squeeze().sum(dim=-1)
        _, h_enc = self.vae.encoder(programs, program_lens)
        z = h_enc.squeeze() if self.vae._vanilla_ae else self.vae._sample_latent(h_enc.squeeze())
        pre_tanh_value = None
        if self._tanh_after_sample or not self.use_decoder_dist:
            pre_tanh_value = z
            z = self.program_vae.vae.tanh(z)

        """ decoder forward pass """
        output = self.vae.decoder(programs, z, teacher_enforcing=evaluate, action=action,
                                  output_mask_all=output_mask_all, eop_action=eop_action, deterministic=deterministic,
                                  evaluate=evaluate)

        value, pred_programs, pred_programs_len, pred_programs_log_probs, output_logits, eop_pred_programs,\
        eop_output_logits, pred_program_masks, dist_entropy = output

        """ Condition policy rollout using sampled latent vector """
        if self.condition_policy.setup == 'RL' and self.use_condition_policy_in_rl:
            agent_value, agent_actions, agent_action_log_probs, agent_action_logits, agent_action_masks, \
            agent_action_dist_entropy = self.condition_policy(init_states, None, z,
                                                                          teacher_enforcing=evaluate,
                                                                          eval_actions=agent_actions,
                                                                          eval_masks_all=agent_action_masks,
                                                                          deterministic=deterministic,
                                                                          evaluate=evaluate)
        else:
            batch_size = z.shape[0]
            agent_value = torch.zeros((batch_size, self.num_demo_per_program, 1), device=z.device, dtype=torch.long)
            agent_actions = torch.zeros((batch_size, self.num_demo_per_program, self.max_demo_length - 1), device=z.device, dtype=torch.long)
            agent_action_log_probs = torch.zeros((batch_size, self.num_demo_per_program, 1), device=z.device, dtype=torch.float)
            agent_action_masks = torch.zeros((batch_size, self.num_demo_per_program, self.max_demo_length - 1), device=z.device, dtype=torch.bool)
            agent_action_dist_entropy = torch.zeros(1, device=z.device, dtype=torch.float)


        """ calculate latent log probs """
        distribution_params = torch.stack((self.vae.z_mean, self.vae.z_sigma), dim=1)
        if not self.use_decoder_dist:
            latent_log_probs = self.vae.dist.log_probs(z, pre_tanh_value)
            latent_dist_entropy = self.vae.dist.normal.entropy().mean()
        else:
            latent_log_probs, latent_dist_entropy = pred_programs_log_probs, dist_entropy

        return value, pred_programs, pred_programs_log_probs, z, pred_program_masks, eop_pred_programs,\
                agent_value, agent_actions, agent_action_log_probs, agent_action_masks, z, distribution_params, dist_entropy,\
                agent_action_dist_entropy, latent_log_probs, latent_dist_entropy

    def _debug_rl_pipeline(self, debug_input):
        for i, idx in enumerate(debug_input['ids']):
            current_update = "_".join(idx.split('_')[:-1])
            for key in debug_input.keys():
                program_idx = int(idx.split('_')[-1])
                act_program_info = self._debug['act'][current_update][key][program_idx]
                if key == 'ids':
                    assert (act_program_info == debug_input[key][i])
                elif 'agent' in key:
                    assert (act_program_info == debug_input[key].view(-1,act_program_info.shape[0] ,debug_input[key].shape[-1])[i]).all()
                else:
                    assert (act_program_info == debug_input[key][i]).all()

    def act(self, programs, rnn_hxs, masks, init_states, deterministic=False):
        program_masks = programs != self.num_program_tokens-1
        outputs = self(programs.long(), program_masks, init_states, None, rnn_hxs, masks, deterministic=deterministic,
                       evaluate=False)
        return outputs

    def get_value(self, programs, rnn_hxs, masks, init_states, deterministic=False):
        program_masks = programs != self.num_program_tokens-1
        outputs = self(programs.long(), program_masks, init_states, None, rnn_hxs, masks, deterministic=deterministic,
                       evaluate=False)

        value, pred_programs, pred_programs_log_probs, z, pred_program_masks, eop_pred_programs, \
        agent_value, agent_actions, agent_action_log_probs, agent_action_masks, z, distribution_params, dist_entropy, \
        agent_action_dist_entropy, latent_log_probs, latent_dist_entropy = outputs
        return value, agent_value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action, output_mask_all, eop_action, agent_actions,
                         agent_action_masks, program_ids, deterministic=False):
        programs, init_states, z = inputs
        program_masks = programs != self.num_program_tokens - 1

        if self._debug:
            self._debug_rl_pipeline({'pred_programs': action,
                                     'pred_program_masks': output_mask_all,
                                     'agent_actions': agent_actions,
                                     'agent_action_masks': agent_action_masks,
                                     'ids': program_ids})

        outputs = self(programs.long(), program_masks, init_states, None, rnn_hxs=rnn_hxs, masks=masks,
                       action=action.long(), output_mask_all=output_mask_all, eop_action=eop_action,
                       agent_actions=agent_actions, agent_action_masks=agent_action_masks,
                       deterministic=deterministic, evaluate=True)
        value, _, pred_programs_log_probs, z, pred_program_masks, _, agent_value, _, agent_action_log_probs, \
        _, _, distribution_params, dist_entropy, agent_action_dist_entropy, latent_log_probs, \
        latent_dist_entropy = outputs

        return value, pred_programs_log_probs, dist_entropy, z, pred_program_masks, agent_value, agent_action_log_probs, \
               agent_action_dist_entropy, distribution_params, latent_log_probs, latent_dist_entropy
