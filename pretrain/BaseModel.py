import os
import time
import pickle
import shutil
from collections import defaultdict
import numpy as np

import torch
import torch.nn as nn
from torch.optim import lr_scheduler

from rl.utils import get_vec_normalize
from pretrain.misc_utils import log_record_dict

optim_list = {
    'sgd': torch.optim.SGD,
    'adagrad': torch.optim.Adagrad,
    'adadelta': torch.optim.Adadelta,
    'adam': torch.optim.Adam,
    'adamax': torch.optim.Adamax,
    'rmsprop': torch.optim.RMSprop,
}


def dfs_freeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)


class BaseModel(object):

    def __init__(self, Net, device, config, envs, dsl, logger, writer, global_logs, verbose):
        self.device = device
        self.config = config
        self.global_logs = global_logs
        self.verbose = verbose
        self.logger = logger
        self.writer = writer
        self.envs = envs
        self.dsl = dsl

        # build policy network
        self.net = Net(envs, **config)
        self.net.to(device)

        # set number of program tokens
        self.num_program_tokens = self.net.num_program_tokens

        # Load parameters if available
        ckpt_path = config['net']['saved_params_path']
        if ckpt_path is not None:
            self.load_net(ckpt_path)

        # disable some parts of network if don't want to train them
        if config['net']['decoder']['freeze_params']:
            assert config['algorithm'] != 'supervisedRL'
            dfs_freeze(self.net.vae.decoder)

        # Initialize optimizer
        self.setup_optimizer(self.net.parameters())

        # Initialize learning rate scheduler
        self.setup_lr_scheduler()

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')

        # Initialize epoch number
        self.epoch = 0

    # FIXME: implement gradien clipping
    def setup_optimizer(self, parameters):
        self.optimizer = None
        if 'optimizer' in self.config:
            optim = optim_list[self.config['optimizer']['name']]
            self.optimizer = optim(filter(lambda p: p.requires_grad, parameters), **self.config['optimizer']['params'])

    def setup_lr_scheduler(self):
        self.scheduler = None
        if self.config['optimizer'].get('scheduler'):
            self.scheduler = lr_scheduler.StepLR(self.optimizer, **self.config['optimizer']['scheduler'])
            self.logger.debug('Using LR scheduler: '+ str(self.config['optimizer']['scheduler']))

    def step_lr_scheduler(self):
        if self.scheduler:
            self.scheduler.step()
            self.logger.debug("Learning rate: %s" % (','.join([str(lr) for lr in self.scheduler.get_lr()])))

    def _add_program_latent_vectors(self, optional_record_dict, optional_record_dict_eval, type='best'):
        self.global_logs['info']['logs']['validation'][type + '_program_latent_vectors'] = optional_record_dict_eval[
            'program_latent_vectors']
        self.global_logs['info']['logs']['validation'][type + '_program_ids'] = optional_record_dict_eval['program_ids']

        self.global_logs['info']['logs']['train'][type + '_program_latent_vectors'] = optional_record_dict[
            'program_latent_vectors']
        self.global_logs['info']['logs']['train'][type+'_program_ids'] = optional_record_dict['program_ids']

    def _run_epoch(self, data_loader, mode, epoch, *args, **kwargs):
        epoch_info = {}
        optinal_epoch_info = {}
        num_batches = len(data_loader)

        batch_info_list = defaultdict(list)
        batch_gt_programs, batch_pred_programs, batch_gen_programs = [], [], []
        batch_program_ids, batch_latent_programs = [], []
        for batch_idx, batch in enumerate(data_loader):

            batch_info = self._run_batch(batch, mode)

            # log losses and accuracies
            for key, val in batch_info.items():
                if 'loss' in key or 'accuracy' in key:
                    batch_info_list[key].append(val)
                    vtype = 'loss' if 'loss' in key else 'accuracy'
                    self.writer.add_scalar('{}_{}/batch_{}'.format(mode, vtype, key), val,
                                           epoch * num_batches + batch_idx)
            # log programs
            batch_gt_programs.append(batch_info['gt_programs'])
            batch_pred_programs.append(batch_info['pred_programs'])
            batch_program_ids.append(batch_info['program_ids'])
            batch_latent_programs.append(batch_info['latent_vectors'])

            self.logger.debug("epoch:{} batch:{}/{} current batch loss: {}".format(epoch, batch_idx, num_batches,
                                                                                   batch_info['total_loss']))
            if mode == 'eval':
                for i in range(min(batch_info['gt_programs'].shape[0], 5)):
                    self.writer.add_text('dataset/epoch_{}'.format(epoch),
                                         'gt: {} pred: {}'.format(self.dsl.intseq2str(batch_info['gt_programs'][i]),
                                                                  self.dsl.intseq2str(batch_info['pred_programs'][i])),
                                         epoch * num_batches)

                batch_gen_programs.append(batch_info['generated_programs'])
                for i, program in enumerate(batch_info['generated_programs']):
                    self.writer.add_text('generated/epoch_{}'.format(epoch), program, epoch * num_batches)

        epoch_info['generated_programs'] = batch_gen_programs
        optinal_epoch_info['program_ids'] = batch_program_ids
        optinal_epoch_info['program_latent_vectors'] = batch_latent_programs
        for key, val in batch_info_list.items():
            if 'loss' in key or 'accuracy' in key:
                vtype = 'loss' if 'loss' in key else 'accuracy'
                epoch_info['mean_'+key] = np.mean(np.array(val).flatten())
                self.writer.add_scalar('{}_{}/epoch_{}'.format(mode, vtype, key), epoch_info['mean_'+key], epoch)
        return epoch_info, optinal_epoch_info

    def run_one_epoch(self, epoch, best_valid_epoch, best_valid_loss, tr_loader, val_loader, *args, **kwargs):
        self.logger.debug('\n' + 40 * '%' + '    EPOCH {}   '.format(epoch) + 40 * '%')
        self.epoch = epoch

        # Run train epoch
        t = time.time()
        record_dict, optional_record_dict = self._run_epoch(tr_loader, 'train', epoch, *args, **kwargs)

        # log all items in dict
        log_record_dict('train', record_dict, self.global_logs)
        # produce print-out
        if self.verbose:
            self._print_record_dict(record_dict, 'train', time.time() - t)

        if val_loader is not None:
            # Run valid epoch
            t = time.time()
            record_dict_eval, optional_record_dict_eval = self._run_epoch(val_loader, 'eval', epoch,
                                                                          self.config['valid']['debug_samples'],
                                                                          'valid_e{}'.format(epoch),
                                                                          batch_size=self.config['valid']['batch_size'])

            # add logs
            log_record_dict('validation', record_dict_eval, self.global_logs)

            # produce print-out
            if self.verbose:
                self._print_record_dict(record_dict_eval, 'validation', time.time() - t)

            if record_dict_eval['mean_total_loss'] < best_valid_loss:
                best_valid_epoch = epoch
                best_valid_loss = record_dict_eval['mean_total_loss']
                self._add_program_latent_vectors(optional_record_dict, optional_record_dict_eval, type='best')
                self.save_net(
                    os.path.abspath(os.path.join(self.config['outdir'], 'best_valid_params.ptp'.format(epoch))))

            if np.isnan(record_dict_eval['mean_total_loss']):
                self.logger.debug(self.verbose, 'Early Stopping because validation loss is nan')
                return best_valid_epoch, best_valid_loss, True

        # Perform LR scheduler step
        self.step_lr_scheduler()

        # Save net
        self._add_program_latent_vectors(optional_record_dict, optional_record_dict_eval, type='final')
        self.save_net(os.path.join(self.config['outdir'], 'final_params.ptp'))

        # Save results
        pickle.dump(self.global_logs, file=open(os.path.join(self.config['outdir'], self.config['record_file']), 'wb'))

        return best_valid_epoch, best_valid_loss, record_dict_eval, False

    def train(self,  train_dataloader, val_dataloader, *args, **kwargs):
        tr_loader = train_dataloader
        val_loader = val_dataloader

        # Initialize params
        max_epoch = kwargs['max_epoch']

        # Train epochs
        best_valid_loss = np.inf
        best_valid_epoch = 0

        for epoch in range(max_epoch):
            best_valid_epoch, best_valid_loss, record_dict_eval, done = self.run_one_epoch(epoch, best_valid_epoch,
                                                                                           best_valid_loss, tr_loader,
                                                                                           val_loader, *args, **kwargs)
            assert not done, 'found NaN in parameters'

        return None

    def evaluate(self, data_loader, epoch=0, *args, **kwargs):
        t = time.time()

        if self.config['mode'] == 'eval':
            assert self.config['net'][
                       'saved_params_path'] is not None, 'need trained parameters to evaluate, got {}'.format(
                self.config['net']['saved_params_path'])

        epoch_records, _ = self._run_epoch(data_loader, 'eval', epoch, *args, **kwargs)

        # Log and print epoch records
        log_record_dict('eval', epoch_records, self.global_logs)
        self._print_record_dict(epoch_records, 'Eval', time.time() - t)
        self.global_logs['result'].update({
            'loss': epoch_records['mean_total_loss'],
        })

        # Save results
        pickle.dump(self.global_logs, file=open(
            os.path.join(self.config['outdir'], self.config['record_file'].replace('.pkl', '_eval.pkl')), 'wb'))

    def save_net(self, filename):
        params = [self.net.state_dict(), getattr(get_vec_normalize(self.envs), 'ob_rms', None)]
        torch.save(params, filename)
        self.logger.debug('params saved to {}'.format(filename))

    def load_net(self, filename):
        self.logger.debug('Loading params from {}'.format(filename))
        params = torch.load(filename, map_location=self.device)
        self.net.load_state_dict(params[0], strict=False)

    def _print_record_dict(self, record_dict, usage, t_taken):
        loss_str = ''
        for k, v in record_dict.items():
            if 'loss' not in k and 'accuracy' not in k:
                continue
            loss_str = loss_str + ' {}: {:.4f}'.format(k, v)

        self.logger.debug('{}:{} took {:.3f}s'.format(usage, loss_str, t_taken))
        return None
