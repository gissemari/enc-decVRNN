import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .decoder import Decoder
from .decoderVRNN import VRNN
from .encoder import Encoder

from selfModules.embedding import Embedding

from utils.functional import kld_coef, parameters_allocation_check, fold,kl_anneal_function


class RVAE(nn.Module):
    def __init__(self, params, use_VRNN):
        super(RVAE, self).__init__()

        self.params = params

        self.embedding = Embedding(self.params, '')

        self.encoder = Encoder(self.params)

        self.context_to_mu = nn.Linear(self.params.encoder_rnn_size * 2, self.params.latent_variable_size)
        self.context_to_logvar = nn.Linear(self.params.encoder_rnn_size * 2, self.params.latent_variable_size)

        if use_VRNN:
            print ("Using DecVRNN")
            self.decoder = VRNN(self.params.word_embed_size, self.params.decoder_rnn_size, self.params.latent_variable_size, self.params.decoder_num_layers, self.params.word_vocab_size) #x_dim, h_dim, z_dim, n_layers
        else:
            self.decoder = Decoder(self.params)

    def forward(self, drop_prob,
                encoder_word_input=None, encoder_character_input=None,
                decoder_word_input=None, decoder_character_input=None,
                z=None, initial_state=None):
        """
        :param encoder_word_input: An tensor with shape of [batch_size, seq_len] of Long type
        :param encoder_character_input: An tensor with shape of [batch_size, seq_len, max_word_len] of Long type
        :param decoder_word_input: An tensor with shape of [batch_size, max_seq_len + 1] of Long type
        :param initial_state: initial state of decoder rnn in order to perform sampling

        :param drop_prob: probability of an element of decoder input to be zeroed in sense of dropout

        :param z: context if sampling is performing

        :return: unnormalized logits of sentence words distribution probabilities
                    with shape of [batch_size, seq_len, word_vocab_size]
                 final rnn state with shape of [num_layers, batch_size, decoder_rnn_size]
        """

        #assert parameters_allocation_check(self), 'Invalid CUDA options. Parameters should be allocated in the same memory'
        use_cuda = self.embedding.word_embed.weight.is_cuda

        assert z is None and fold(lambda acc, parameter: acc and parameter is not None,
                                  [encoder_word_input, encoder_character_input, decoder_word_input],
                                  True) \
            or (z is not None and decoder_word_input is not None), \
            "Invalid input. If z is None then encoder and decoder inputs should be passed as arguments"

        if z is None:
            ''' Get context from encoder and sample z ~ N(mu, std)
            '''
            [batch_size, _] = encoder_word_input.size()

            encoder_input = self.embedding(encoder_word_input, encoder_character_input)

            context = self.encoder(encoder_input) #final state

            mu = self.context_to_mu(context)
            logvar = self.context_to_logvar(context) # to z sampled from 
            std = t.exp(0.5 * logvar)

            z = Variable(t.randn([batch_size, self.params.latent_variable_size]))
            if use_cuda:
                z = z.cuda()

            z = z * std + mu

            # sentence-VAE 
            kld = -0.5 * t.sum(1 + logvar - t.pow(mu,2) - t.exp(logvar))
            #kld = (-0.5 * t.sum(logvar - t.pow(mu, 2) - t.exp(logvar) + 1, 1)).mean().squeeze()
        else:
            kld = None

        decoder_input = self.embedding.word_embed(decoder_word_input)

        kld_loss, nll_loss, (all_enc_mean, all_enc_std), (all_dec_mean, all_dec_std), out, final_state = self.decoder(decoder_input, z, drop_prob, initial_state)
        # zeroes some of the elements of the input tensor with probability p

        return out, final_state, kld

    def learnable_parameters(self):

        # word_embedding is constant parameter thus it must be dropped from list of parameters for optimizer
        return [p for p in self.parameters() if p.requires_grad]

    def trainer(self, optimizer, batch_loader):
        def train(i, batch_size, use_cuda, dropout):
            input = batch_loader.next_batch(batch_size, 'train')
            input = [Variable(t.from_numpy(var)) for var in input]
            input = [var.long() for var in input]
            input = [var.cuda() if use_cuda else var for var in input]

            [encoder_word_input, encoder_character_input, decoder_word_input, decoder_character_input, target] = input

            logits, _, kld = self(dropout,
                                  encoder_word_input, encoder_character_input,
                                  decoder_word_input, decoder_character_input,
                                  z=None)

            logits = logits.view(-1, self.params.word_vocab_size)
            target = target.view(-1)
            cross_entropy = F.cross_entropy(logits, target,size_average=False) #gisse: added size_average
            '''
            # loss calculation
            NLL_loss, KL_loss, KL_weight = loss_fn(logp, batch['target'],
            batch['length'], mean, logv, args.anneal_function, step, args.k, args.x0)

            loss = (NLL_loss + KL_weight * KL_loss)/batch_size
            '''
            kldWeight = kld_coef(i)# kl_anneal_function(i)
            loss =  (cross_entropy +  kldWeight* kld)/batch_size #if taken out (79), it vanishes fast -> kl annealing. taken back:79 * cross_entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            return cross_entropy, kld, kldWeight, loss# kld_coef(i) #also Sentence-VAE does for iteration

        return train

    def validater(self, batch_loader):
        def validate(i,batch_size, use_cuda, instance):#i - iteration
            input = batch_loader.next_batch(batch_size, 'valid')
            input = [Variable(t.from_numpy(var)) for var in input]
            input = [var.long() for var in input]
            input = [var.cuda() if use_cuda else var for var in input]

            [encoder_word_input, encoder_character_input, decoder_word_input, decoder_character_input, target] = input

            logits, _, kld = self(0., #none input is converted to 0
                                  encoder_word_input, encoder_character_input,
                                  decoder_word_input, decoder_character_input,
                                  z=None)

            logitsSoft = F.softmax(logits,dim=2) #Gissella added
            logitsSoft = logitsSoft.view(-1, self.params.word_vocab_size)
            logits = logits.view(-1, self.params.word_vocab_size)
            target = target.view(-1)

            cross_entropy = F.cross_entropy(logits, target,size_average=False)

            kldWeight = kl_anneal_function(i)
            loss =  (cross_entropy +  kldWeight* kld)/batch_size

            return cross_entropy, kld, logitsSoft, loss, encoder_word_input[instance]

        return validate

    def sample(self, batch_loader, seq_len, seed, use_cuda):
        seed = Variable(t.from_numpy(seed).float())
        if use_cuda:
            seed = seed.cuda()

        #batchsize=1
        decoder_word_input_np, decoder_character_input_np = batch_loader.go_input(1) #[[self.word_to_idx[self.go_token]] for _ in range(batch_size)]

        decoder_word_input = Variable(t.from_numpy(decoder_word_input_np).long())
        decoder_character_input = Variable(t.from_numpy(decoder_character_input_np).long())

        if use_cuda:
            decoder_word_input, decoder_character_input = decoder_word_input.cuda(), decoder_character_input.cuda()

        result = ''

        initial_state = None

        for i in range(seq_len):
            logits, initial_state, _ = self(0., None, None,
                                            decoder_word_input, decoder_character_input,
                                            seed, initial_state)

            logits = logits.view(-1, self.params.word_vocab_size)
            prediction = F.softmax(logits)

            word = batch_loader.sample_word_from_distribution(prediction.data.cpu().numpy()[-1])

            if word == batch_loader.end_token:
                break

            result += ' ' + word

            decoder_word_input_np = np.array([[batch_loader.word_to_idx[word]]]) # feeding previous 
            decoder_character_input_np = np.array([[batch_loader.encode_characters(word)]])

            decoder_word_input = Variable(t.from_numpy(decoder_word_input_np).long())
            decoder_character_input = Variable(t.from_numpy(decoder_character_input_np).long())

            if use_cuda:
                decoder_word_input, decoder_character_input = decoder_word_input.cuda(), decoder_character_input.cuda()

        return result
