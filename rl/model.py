import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.utils import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size, rnn_type='GRU'):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent
        self.rnn_type = rnn_type

        if recurrent:
            if rnn_type == 'GRU':
                self.gru = nn.GRU(recurrent_input_size, hidden_size)
            elif rnn_type == 'LSTM':
                self.lstm = nn.LSTM(recurrent_input_size, hidden_size)
                # need to keep this for backward compatibility to pre-trained weights
                self.gru = self.lstm
            else:
                raise NotImplementedError()

            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_rnn(self, x, hxs, masks):
        if self.rnn_type == 'GRU':
            return self._forward_gru(x, hxs, masks)
        elif self.rnn_type == 'LSTM':
            return self._forward_lstm(x, hxs, masks)
        else:
            raise NotImplementedError()

    def _forward_lstm(self, x, hxs, masks):
        assert isinstance(hxs, tuple) and len(hxs) == 2
        if x.size(0) == hxs[0].size(0):
            x, hxs = self.lstm(x.unsqueeze(0), (hxs[0].unsqueeze(0), hxs[1].unsqueeze(0)))
            x = x.squeeze(0)
            hxs = (hxs[0].squeeze(0), hxs[1].squeeze(0))
        else:
            assert 0, "current code doesn't support this"
        return x, hxs

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            assert 0, "current code doesn't support this"
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs