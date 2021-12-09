import sys
import time
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from pretrain.BaseRLModel import BaseRLModel
from pretrain.models import VAE, ConditionPolicy, ProgramVAE
from rl.distributions import FixedNormal
from rl.utils import masked_mean, init

class NoiseInjection(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(channel,))

    def forward(self, image, noise):
        return image + self.weight * noise


""" Meta Controller to search over intention space """
class MetaController(nn.Module):
    def __init__(self, num_outputs, add_noise=False):
        super(MetaController, self).__init__()
        self.add_noise = add_noise

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        if self.add_noise:
            self.noise0 = NoiseInjection(num_outputs)
            self.layer1 = init_(nn.Linear(num_outputs, num_outputs))
            self.noise1 = NoiseInjection(num_outputs)
            self.tanh1 = nn.Tanh()
            self.layer2 = init_(nn.Linear(num_outputs, num_outputs))
            self.noise2 = NoiseInjection(num_outputs)
            self.tanh2 = nn.Tanh()
        else:
            self.meta_controller = nn.Sequential(init_(nn.Linear(num_outputs, num_outputs)), nn.Tanh(),
                                                 init_(nn.Linear(num_outputs, num_outputs)), nn.Tanh())

    def forward(self, inputs, noise=None):

        if self.add_noise:
            if noise is None:
                noise = []
                for i in range(3):
                    noise.append(1 * torch.randn_like(inputs))

            out = self.noise0(inputs, noise[0])
            out = self.layer1(out)
            out = self.tanh1(self.noise1(out, noise[1]))
            out = self.layer2(out)
            out = self.tanh2(self.noise2(out, noise[2]))
            return out
        else:
            return self.meta_controller(inputs)


""" RL Architecture """
class ProgramRLVAE(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ProgramRLVAE, self).__init__()
        self.meta_controller = MetaController(kwargs['num_lstm_cell_units'],
                                              add_noise=kwargs['net']['controller']['add_noise'])
        self.program_vae = ProgramVAE(*args, **kwargs)
        self.num_program_tokens = self.program_vae.num_program_tokens
        self.distribution_params = None
        self.controller_input_coef = kwargs['net']['controller']['input_coef']
        self._tanh_after_sample = kwargs['net']['tanh_after_sample']
        self._debug = kwargs['debug']
        self.use_decoder_dist = kwargs['net']['controller']['use_decoder_dist']
        self.use_condition_policy_in_rl = kwargs['rl']['loss']['condition_rl_loss_coef'] > 0.0
        self.num_demo_per_program = kwargs['rl']['envs']['executable']['num_demo_per_program']
        self.max_demo_length = kwargs['rl']['envs']['executable']['max_demo_length']
        self.use_previous_programs = kwargs['net']['controller']['use_previous_programs']
        self.program_reduction = kwargs['net']['controller']['program_reduction']
        self.use_all_programs = kwargs['rl']['use_all_programs']

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.program_vae.vae.latent_dim

    @property
    def is_recurrent(self):
        return self.program_vae.vae.encoder.is_recurrent

    def _debug_rl_pipeline(self, debug_input):
        for i, idx in enumerate(debug_input['ids']):
            current_update = "_".join(idx.split('_')[:-1])
            for key in debug_input.keys():
                program_idx = int(idx.split('_')[-1])
                act_program_info = self._debug['act'][current_update][key][program_idx]
                if key == 'ids':
                    assert (act_program_info == debug_input[key][i])
                else:
                    assert (act_program_info == debug_input[key][i]).all()

    def forward(self, h_meta, policy_inputs, rnn_hxs, masks, action=None, output_mask_all=None, eop_action=None,
                agent_actions=None, agent_action_masks=None, deterministic=False, evaluate=False):
        decoder_inputs, condition_inputs = policy_inputs

        """ Sample latent vector from the output of meta controller """
        z = self.program_vae.vae._sample_latent(h_meta.squeeze())
        pre_tanh_value = None
        if self._tanh_after_sample or not self.use_decoder_dist:
            pre_tanh_value = z
            z = self.program_vae.vae.tanh(z)
        distribution_params = torch.stack((self.program_vae.vae.z_mean, self.program_vae.vae.z_sigma), dim=1)
        if not self.use_decoder_dist:
            latent_log_probs = self.program_vae.vae.dist.log_probs(z, pre_tanh_value)
            latent_dist_entropy = self.program_vae.vae.dist.normal.entropy().mean()

        """ decode sampled latent vector to a program """
        output = self.program_vae.vae.decoder(None, z, teacher_enforcing=evaluate, action=action,
                                              output_mask_all=output_mask_all, eop_action=eop_action,
                                              deterministic=deterministic, evaluate=evaluate)

        value, pred_programs, pred_programs_len, pred_programs_log_probs, output_logits, eop_pred_programs,\
        eop_output_logits, pred_program_masks, dist_entropy = output
        # FIXME: hack to ignore calculating latent distributions
        if self.use_decoder_dist: latent_log_probs, latent_dist_entropy = pred_programs_log_probs, dist_entropy

        """ Condition policy rollout using sampled latent vector """
        if self.program_vae.condition_policy.setup == 'RL' and self.use_condition_policy_in_rl:
            agent_value, agent_actions, agent_action_log_probs, agent_action_logits, agent_action_masks,\
            agent_action_dist_entropy = self.program_vae.condition_policy(condition_inputs, None, z,
                                                                          teacher_enforcing=evaluate,
                                                                          eval_actions=agent_actions,
                                                                          eval_masks_all=agent_action_masks,
                                                                          evaluate=evaluate)
        else:
            batch_size = z.shape[0]
            agent_value = torch.zeros((batch_size, self.num_demo_per_program, 1), device=z.device, dtype=torch.long)
            agent_actions = torch.zeros((batch_size, self.num_demo_per_program, self.max_demo_length - 1), device=z.device, dtype=torch.long)
            agent_action_log_probs = torch.zeros((batch_size, self.num_demo_per_program, 1), device=z.device, dtype=torch.float)
            agent_action_masks = torch.zeros((batch_size, self.num_demo_per_program, self.max_demo_length - 1), device=z.device, dtype=torch.bool)
            agent_action_dist_entropy = torch.zeros(1, device=z.device, dtype=torch.float)


        return value, pred_programs, pred_programs_log_probs, z, pred_program_masks, eop_pred_programs, agent_value, \
               agent_actions, agent_action_log_probs, agent_action_masks, z, distribution_params, dist_entropy, \
               agent_action_dist_entropy, latent_log_probs, latent_dist_entropy

    def act(self, inputs, rnn_hxs, masks, condition_inputs, deterministic=False):
        meta_inputs = self.controller_input_coef * torch.ones_like(rnn_hxs)
        if self.use_previous_programs:
            input_masks = masks.repeat(1, rnn_hxs.shape[1]).to(torch.bool)
            mean_rnn_hxs = rnn_hxs.mean(0, True)
            mean_rnn_hxs = mean_rnn_hxs.expand(rnn_hxs.shape[0], -1)
            if not self.use_all_programs:
                assert input_masks.all() == True
            prev_program_input = rnn_hxs if self.program_reduction == 'identity' else mean_rnn_hxs
            meta_inputs = torch.where(input_masks, prev_program_input, meta_inputs)
        h_meta = self.meta_controller(meta_inputs)
            
        policy_inputs = (None, condition_inputs)
        outputs = self(h_meta, policy_inputs, rnn_hxs, masks, deterministic=deterministic, evaluate=False)
        return outputs

    def get_value(self, inputs, rnn_hxs, masks, condition_inputs, deterministic=False):
        outputs = self.act(inputs, rnn_hxs, masks, condition_inputs, deterministic=deterministic)

        value, pred_programs, pred_programs_log_probs, z, pred_program_masks, eop_pred_programs, agent_value, \
        agent_actions, agent_action_log_probs, agent_action_masks, z, distribution_params, dist_entropy, \
        agent_action_dist_entropy, latent_log_probs, latent_dist_entropy = outputs
        return value, agent_value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action, output_mask_all, eop_action, agent_actions,
                         agent_action_masks, program_ids, deterministic=False):
        inputs, condition_inputs, z = inputs
        policy_inputs = (None, condition_inputs)

        if self._debug:
            self._debug_rl_pipeline({'pred_programs': action,
                                    'pred_program_masks': output_mask_all,
                                    'agent_actions': agent_actions,
                                    'agent_action_masks': agent_action_masks,
                                    'ids': program_ids})

        meta_inputs = self.controller_input_coef * torch.ones_like(rnn_hxs)
        if self.use_previous_programs:
            input_masks = masks.repeat(1, rnn_hxs.shape[1]).to(torch.bool)
            mean_rnn_hxs = rnn_hxs.mean(0, True)
            mean_rnn_hxs = mean_rnn_hxs.expand(rnn_hxs.shape[0], -1)
            if not self.use_all_programs:
                assert input_masks.all() == True
            prev_program_input = rnn_hxs if self.program_reduction == 'identity' else mean_rnn_hxs
            meta_inputs = torch.where(input_masks, rnn_hxs, meta_inputs)
        h_meta = self.meta_controller(meta_inputs)
        outputs = self(h_meta, policy_inputs, rnn_hxs, masks, action.long(), output_mask_all, eop_action, agent_actions,
                       agent_action_masks, deterministic=deterministic, evaluate=True)
        value, _, pred_programs_log_probs, z, pred_program_masks, _, agent_value, _, agent_action_log_probs,\
        _, _, distribution_params, dist_entropy, agent_action_dist_entropy, latent_log_probs,\
        latent_dist_entropy = outputs

        return value, pred_programs_log_probs, dist_entropy, z, pred_program_masks, agent_value, agent_action_log_probs,\
               agent_action_dist_entropy, distribution_params, latent_log_probs, latent_dist_entropy

class RLModel(BaseRLModel):
    def __init__(self, *args, **kwargs):
        super(RLModel, self).__init__(ProgramRLVAE, None, *args, **kwargs)
        self._two_head = self.config['rl']['policy']['two_head']
        self.max_demo_length = self.config['rl']['envs']['executable']['max_demo_length']
        self.latent_loss_coef = self.config['loss']['latent_loss_coef']
        self.condition_loss_coef = self.config['loss']['condition_loss_coef']

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.net.recurrent_hidden_state_size


class RLModel2(BaseRLModel):
    def __init__(self, program_vae_net, *args, **kwargs):
        super(RLModel2, self).__init__(None, program_vae_net, *args, **kwargs)
        self._two_head = self.config['rl']['policy']['two_head']
        self.max_demo_length = self.config['rl']['envs']['executable']['max_demo_length']
        self.latent_loss_coef = self.config['loss']['latent_loss_coef']
        self.condition_loss_coef = self.config['loss']['condition_loss_coef']

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.net.recurrent_hidden_state_size
