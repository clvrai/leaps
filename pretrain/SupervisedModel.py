import sys
import copy
import time
import numpy as np
import torch
import torch.nn as nn

from pretrain.BaseModel import BaseModel
from pretrain.models import VAE, ConditionPolicy, ProgramVAE
from rl.utils import masked_mean


def calculate_accuracy(logits, targets, mask, batch_shape):
    masked_preds = (logits.argmax(dim=-1, keepdim=True) * mask).view(*batch_shape, 1)
    masked_targets = (targets * mask).view(*batch_shape, 1)
    t_accuracy = 100 * masked_mean((masked_preds == masked_targets).float(), mask.view(*masked_preds.shape),
                                   dim=1).mean()

    p_accuracy = 100 * (masked_preds.squeeze() == masked_targets.squeeze()).all(dim=1).float().mean()
    return t_accuracy, p_accuracy


class SupervisedModel(BaseModel):
    def __init__(self, *args, **kwargs):
        super(SupervisedModel, self).__init__(ProgramVAE, *args, **kwargs)
        self._two_head = self.config['two_head']
        self.max_demo_length = self.config['max_demo_length']
        self.latent_loss_coef = self.config['loss']['latent_loss_coef']
        self.condition_loss_coef = self.config['loss']['condition_loss_coef']
        self._vanilla_ae = self.config['AE']
        self._disable_decoder = self.config['net']['decoder']['freeze_params']
        self._disable_condition = self.config['net']['condition']['freeze_params']
        self.condition_states_source = self.config['net']['condition']['observations']

        # debug code
        self._debug = self.config['debug']
        if self._debug:
            self.debug_dict_act = {}
            self.debug_dict_eval_act = {}

    @property
    def is_recurrent(self):
        return self.net.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.net.base.recurrent_hidden_state_size

    def _get_condition_loss(self, a_h, a_h_len, action_logits, action_masks):
        """ loss between ground truth trajectories and predicted action sequences

        :param a_h(int16): B x num_demo_per_program x max_demo_length
        :param a_h_len(int16): a_h_len: B x num_demo_per_program
        :param action_logits: (B * num_demo_per_programs) x max_a_h_len x num_actions
        :param action_masks: (B * num_demo_per_programs) x max_a_h_len x 1
        :return (float): condition policy loss
        """
        batch_size_x_num_demo_per_program, max_a_h_len, num_actions = action_logits.shape
        assert max_a_h_len == a_h.shape[-1]

        padded_preds = action_logits

        """ add dummy logits to targets """
        target_masks = a_h != self.net.condition_policy.num_agent_actions - 1
        # remove temporarily added no-op actions in case of empty trajectory to
        # verify target masks
        a_h_len2 = a_h_len - (a_h[:,:,0] == self.net.condition_policy.num_agent_actions - 1).to(a_h_len.dtype)
        assert (target_masks.sum(dim=-1).squeeze() == a_h_len2.squeeze()).all()
        targets = torch.where(target_masks, a_h, (num_actions-1) * torch.ones_like(a_h))

        """ condition mask """
        # flatten everything and select actions that you want in backpropagation
        target_masks = target_masks.view(-1, 1)
        action_masks = action_masks.view(-1, 1)
        cond_mask = torch.max(action_masks, target_masks)

        # gather prediction that needs backpropagation
        subsampled_targets = targets.view(-1,1)[cond_mask].long()
        subsampled_padded_preds = padded_preds.view(-1, num_actions)[cond_mask.squeeze()]

        condition_loss = self.loss_fn(subsampled_padded_preds, subsampled_targets)

        """ calculate accuracy """
        with torch.no_grad():
            batch_shape = padded_preds.shape[:-1]
            cond_t_accuracy, cond_p_accuracy = calculate_accuracy(padded_preds.view(-1, num_actions),
                                                                  targets.view(-1, 1), cond_mask, batch_shape)

        return condition_loss, cond_t_accuracy, cond_p_accuracy

    def _greedy_rollout(self, batch, z, targets, trg_mask, mode):
        programs, _, _, s_h, a_h, a_h_len = batch

        if mode == 'train' and self.condition_states_source != 'initial_state':
            zero_tensor = torch.tensor([0.0], device=self.device, requires_grad=False)
            return (zero_tensor, zero_tensor, zero_tensor, zero_tensor), None, None

        with torch.no_grad():
            # greedy rollout of decoder
            greedy_outputs = self.net.vae.decoder(programs, z, teacher_enforcing=False, deterministic=True)
            _, _, _, _, greedy_output_logits, _, _, pred_program_masks, _ = greedy_outputs

            """ calculate accuracy """
            logits = greedy_output_logits.view(-1, greedy_output_logits.shape[-1])
            pred_mask = pred_program_masks.view(-1, 1)
            vae_mask = torch.max(pred_mask, trg_mask)
            with torch.no_grad():
                batch_shape = greedy_output_logits.shape[:-1]
                greedy_t_accuracy, greedy_p_accuracy = calculate_accuracy(logits, targets, vae_mask, batch_shape)

            _, _, _, action_logits, action_masks, _ = self.net.condition_policy(s_h, a_h, z, teacher_enforcing=False, deterministic=True)
            _, greedy_a_accuracy, greedy_d_accuracy = self._get_condition_loss(a_h, a_h_len, action_logits,
                                                                               action_masks)

            # 2 random vectors
            generated_programs = None
            if mode == 'eval':
                rand_z = torch.randn((2, z.shape[1])).to(z.dtype).to(z.device)
                generated_outputs = self.net.vae.decoder(None, rand_z, teacher_enforcing=False, deterministic=True)
                generated_programs = [self.dsl.intseq2str(prg) for prg in generated_outputs[1]]

        return (greedy_t_accuracy, greedy_p_accuracy, greedy_a_accuracy, greedy_d_accuracy), generated_programs, logits


    def _run_batch(self, batch, mode='train'):
        """ training on one batch

        :param batch: list of 6 elements:
                      programs(long): ground truth programs: B x max_program_len
                      ids(str): program ids
                      trg_mask(bool): masks for identifying valid tokens in programs: B x max_program_len x 1
                      s_h(bool): B x num_demo_per_program x max_demo_length(currently 1) x C x W x H
                      a_h(int16): B x num_demo_per_program x max_demo_length
                      a_h_len(int16): B x num_demo_per_program

        :param mode(str): execution mode, train or eval
        :return (dict): batch_info containing accuracy, loss and predicitons
        """

        # Do mode-based setup
        if mode == 'train':
            self.net.train()
            torch.set_grad_enabled(True)
        elif mode == 'eval':
            self.net.eval()
            torch.set_grad_enabled(False)

        programs, ids, trg_mask, s_h, a_h, a_h_len = batch

        """ forward pass """
        output = self.net(programs, trg_mask, s_h, a_h, deterministic=True)
        pred_programs, pred_program_lens, output_logits, eop_pred_programs, eop_output_logits, pred_program_masks,\
        action_logits, action_masks, z = output

        """ flatten inputs and outputs for loss calculation """
        # skip first token DEF for loss calculation
        targets = programs[:, 1:].contiguous().view(-1, 1)
        trg_mask = trg_mask[:, 1:].contiguous().view(-1, 1)
        logits = output_logits.view(-1, output_logits.shape[-1])
        pred_mask = pred_program_masks.view(-1, 1)
        # need to penalize shorter and longer predicted programs
        vae_mask = torch.max(pred_mask, trg_mask)

        # Do backprop
        if mode == 'train':
            self.optimizer.zero_grad()

        zero_tensor = torch.tensor([0.0], device=self.device, requires_grad=False)
        lat_loss, rec_loss, condition_loss = zero_tensor, zero_tensor, zero_tensor
        cond_t_accuracy, cond_p_accuracy = zero_tensor, zero_tensor
        if not self._disable_decoder:
            rec_loss = self.loss_fn(logits[vae_mask.squeeze()], (targets[vae_mask.squeeze()]).view(-1))
        if not self._vanilla_ae:
            lat_loss = self.net.vae.latent_loss(self.net.vae.z_mean, self.net.vae.z_sigma)
        if not self._disable_condition:
            condition_loss, cond_t_accuracy, cond_p_accuracy = self._get_condition_loss(a_h, a_h_len, action_logits,
                                                                                        action_masks)

        # total loss
        loss = rec_loss + (self.latent_loss_coef * lat_loss) + (self.condition_loss_coef * condition_loss)

        if mode == 'train':
            loss.backward()
            self.optimizer.step()

        """ calculate accuracy """
        with torch.no_grad():
            batch_shape = output_logits.shape[:-1]
            t_accuracy, p_accuracy = calculate_accuracy(logits, targets, vae_mask, batch_shape)
            greedy_accuracies, generated_programs, glogits = self._greedy_rollout(batch, z, targets, trg_mask, mode)
            greedy_t_accuracy, greedy_p_accuracy, greedy_a_accuracy, greedy_d_accuracy = greedy_accuracies

        batch_info = {
            'decoder_token_accuracy': t_accuracy.detach().cpu().numpy().item(),
            'decoder_program_accuracy': p_accuracy.detach().cpu().numpy().item(),
            'condition_action_accuracy': cond_t_accuracy.detach().cpu().numpy().item(),
            'condition_demo_accuracy': cond_p_accuracy.detach().cpu().numpy().item(),
            'decoder_greedy_token_accuracy': greedy_t_accuracy.detach().cpu().numpy().item(),
            'decoder_greedy_program_accuracy': greedy_p_accuracy.detach().cpu().numpy().item(),
            'condition_greedy_action_accuracy': greedy_a_accuracy.detach().cpu().numpy().item(),
            'condition_greedy_demo_accuracy': greedy_d_accuracy.detach().cpu().numpy().item(),
            'total_loss': loss.detach().cpu().numpy().item(),
            'rec_loss': rec_loss.detach().cpu().numpy().item(),
            'lat_loss': lat_loss.detach().cpu().numpy().item(),
            'condition_loss': condition_loss.detach().cpu().numpy().item(),
            'gt_programs': programs.detach().cpu().numpy(),
            'pred_programs': pred_programs.detach().cpu().numpy(),
            'generated_programs': generated_programs,
            'program_ids': ids,
            'latent_vectors': z.detach().cpu().numpy().tolist()}

        return batch_info
