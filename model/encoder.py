import torch as t
import torch.nn as nn
import torch.nn.functional as F

from selfModules.highway import Highway
from utils.functional import parameters_allocation_check


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()

        self.params = params

        self.hw1 = Highway(self.params.sum_depth + self.params.word_embed_size, 2, F.relu)

        self.rnn = nn.LSTM(input_size=self.params.word_embed_size + self.params.sum_depth,
                           hidden_size=self.params.encoder_rnn_size,
                           num_layers=self.params.encoder_num_layers,
                           batch_first=True)#,
                           #bidirectional=True)

    def forward(self, input, State):
        """
        :param input: [batch_size, seq_len, embed_size] tensor
        :return: context of input sentenses with shape of [batch_size, latent_variable_size]
        """

        [batch_size, seq_len, embed_size] = input.size()

        input = input.view(-1, embed_size)
        input = self.hw1(input)
        input = input.view(batch_size, seq_len, embed_size)

        #assert parameters_allocation_check(self), 'Invalid CUDA options. Parameters should be allocated in the same memory'

        ''' Unfold rnn with zero initial state and get its final state from the last layer
        
        _, (_, final_state) = self.rnn(input, State)

        final_state = final_state.view(self.params.encoder_num_layers, 2, batch_size, self.params.encoder_rnn_size)
        final_state = final_state[-1]
        h_1, h_2 = final_state[0], final_state[1]
        final_state = t.cat([h_1, h_2], 1)

        return final_state
        '''
        _, (transfer_state_1, final_state) = self.rnn(input, State)
        transfer_state_2 = final_state
        
        final_state = final_state.view(self.params.encoder_num_layers,  batch_size, self.params.encoder_rnn_size)#2,bath_size because of bidirectional
        final_state = final_state[-1]
        # for 2 layers
        #h_1, h_2 = final_state[0], final_state[1]
        #final_state = t.cat([h_1, h_2], 0)
        # FOR 1 LAYER
        h_1 = final_state[0]
        final_state = h_1# t.cat([h_1, h_2], 0)

        return final_state, transfer_state_1, transfer_state_2