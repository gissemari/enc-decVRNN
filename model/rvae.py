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
from beam_search import Beam

class RVAE(nn.Module):
    def __init__(self, params, use_VRNN, use_cuda=False):
        super(RVAE, self).__init__()

        self.params = params

        self.embedding = Embedding(self.params, '')

        self.encoder = Encoder(self.params)

        self.context_to_mu = nn.Linear(self.params.encoder_rnn_size , self.params.latent_variable_size)
        self.context_to_logvar = nn.Linear(self.params.encoder_rnn_size , self.params.latent_variable_size)

        if use_VRNN:
            print ("Using DecVRNN")
            self.decoder = VRNN(self.params.word_embed_size, self.params.decoder_rnn_size, self.params.latent_variable_size, self.params.decoder_num_layers, self.params.word_vocab_size, use_cuda=use_cuda) #x_dim, h_dim, z_dim, n_layers
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

        #if z is None:
        ''' Get context from encoder and sample z ~ N(mu, std)
        '''
        [batch_size, _] = encoder_word_input.size()

        encoder_input = self.embedding(encoder_word_input, encoder_character_input)

        context, h_0 , c_0 = self.encoder(encoder_input,None) #final state

        mu = self.context_to_mu(context)
        logvar = self.context_to_logvar(context) # to z sampled from 
        std = t.exp(0.5 * logvar)

        z = Variable(t.randn([batch_size, self.params.latent_variable_size]))
        if use_cuda:
            z = z.cuda()

        z = z * std + mu

        # sentence-VAE 
        kld = -0.5 * t.sum(1 + logvar - t.pow(mu,2) - t.exp(logvar))
        #print("h-c",h_0.shape, context.shape)#4,64,48 - 64,96
        initial_state = [h_0, c_0]
        #print("initial_state in rvae ",initial_state[0].shape)#2,64
        #kld = (-0.5 * t.sum(logvar - t.pow(mu, 2) - t.exp(logvar) + 1, 1)).mean().squeeze()

       # else:
       #     kld = None

        #print("initial_state in rvae 2",initial_state[0].shape)
        decoder_input = self.embedding.word_embed(decoder_word_input)
        kld_local, nll_local, (all_enc_mean, all_enc_std), (all_dec_mean, all_dec_std), out, final_state = self.decoder(decoder_input, z, drop_prob, initial_state[0])
        # zeroes some of the elements of the input tensor with probability p

        return out, final_state,  kld, kld_local, nll_local

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

            logits, _, kld_global, kld_local, nll_local = self(dropout,
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
            #cross_entropy +
            loss =  ( kldWeight* kld_global + kldWeight*kld_local + nll_local/10000)/batch_size #if taken out (79), it vanishes fast -> kl annealing. taken back:79 * cross_entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            return cross_entropy, nll_local, kld_global, kld_local, kldWeight, loss# kld_coef(i) #also Sentence-VAE does for iteration

        return train

    def validater(self, batch_loader):
        def validate(i,batch_size, use_cuda,dropout):#, instance#i - iteration
            input = batch_loader.next_batch(batch_size, 'valid')
            input = [Variable(t.from_numpy(var)) for var in input]
            input = [var.long() for var in input]
            input = [var.cuda() if use_cuda else var for var in input]

            [encoder_word_input, encoder_character_input, decoder_word_input, decoder_character_input, target] = input

            logits, _, kld , kld_local, nll_local= self(dropout, #none input is converted to 0
                                  encoder_word_input, encoder_character_input,
                                  decoder_word_input, decoder_character_input,
                                  z=None)

            logitsSoft = F.softmax(logits,dim=2) #Gissella added
            logitsSoft = logitsSoft.view(-1, self.params.word_vocab_size)
            logits = logits.view(-1, self.params.word_vocab_size)
            target = target.view(-1)

            cross_entropy = F.cross_entropy(logits, target,size_average=False)

            kldWeight = kld_coef(i)#1# kl_anneal_function(i)
            loss =  (cross_entropy +  kldWeight* kld)/batch_size

            # Gissella. to decoder wordEmbedding for output
            seq_len = 20
            result = ''
            print (logitsSoft.shape)
            for i in range(logitsSoft.shape[0]):
                word = batch_loader.sample_word_from_distribution(logitsSoft.data.cpu().numpy()[i])#-1
                if word == batch_loader.end_token:
                    break
                result += ' ' + word

            return cross_entropy, kld, logitsSoft, loss, result# encoder_word_input[instance]

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
            logits, initial_state, _, kld_local, nll_local = self(0., None, None,
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


    ######### Adding this function from paraphrase generation repository
    
    def sampler(self, batch_loader, seq_len, seed, use_cuda,i,beam_size,n_best):
        input = batch_loader.next_batch(1, 'valid')
        input = [Variable(t.from_numpy(var)) for var in input]
        input = [var.long() for var in input]
        input = [var.cuda() if use_cuda else var for var in input]
        [encoder_word_input, encoder_character_input, decoder_word_input, decoder_character_input, target] = input

        encoder_input = self.embedding(encoder_word_input, encoder_character_input)

        _ , h0 , c0 = self.encoder(encoder_input, None)
        State = (h0,c0)

        # print '----------------------'
        # print 'Printing h0 ---------->'
        # print h0
        # print '----------------------'

        # State = None
        results, scores = self.sample_beam(batch_loader, seq_len, seed, use_cuda, State, beam_size, n_best)

        return results, scores

    def sample_beam(self, batch_loader, seq_len, seed, use_cuda, State, beam_size, n_best):
        # seed = Variable(t.from_numpy(seed).float())
        if use_cuda:
            seed = seed.cuda()

        decoder_word_input_np, decoder_character_input_np = batch_loader.go_input(1)

        decoder_word_input = Variable(t.from_numpy(decoder_word_input_np).long())
        decoder_character_input = Variable(t.from_numpy(decoder_character_input_np).long())

        if use_cuda:
            decoder_word_input, decoder_character_input = decoder_word_input.cuda(), decoder_character_input.cuda()

        dec_states = State
        #print ("decStates antes",dec_states[0].shape) #([2, 1, 64] #2 layers, 1 step, 64 units en enc
        '''
        dec_states = [
            dec_states[0].repeat(1, beam_size, 1),
            dec_states[1].repeat(1, beam_size, 1)
        ]
        '''
        dec_states = dec_states[0].repeat(1, beam_size, 1)
        #print ("decStates despues",dec_states[0].shape) #2,5,64
        drop_prob = 0.0
        beam_size = beam_size
        batch_size = 1
        
        beam = [Beam(beam_size, batch_loader, cuda=use_cuda) for k in range(batch_size)]

        batch_idx = list(range(batch_size))
        #print("batch_idx ", batch_idx)
        remaining_sents = batch_size
        
        
        for i in range(seq_len):
            
            input = t.stack([b.get_current_state() for b in beam if not b.done]).t().contiguous().view(1, -1)

            trg_emb = self.embedding.word_embed(Variable(input).transpose(1, 0))
            
            kld_loss, nll_loss, (all_enc_mean, all_enc_std), (all_dec_mean, all_dec_std), trg_h, dec_states =self.decoder(trg_emb, seed, drop_prob, dec_states, sampling=True)
            
            #trg_h, dec_states = self.decoder.only_decoder_beam(trg_emb, seed, drop_prob, dec_states)
            #print("dec_states after decoder fwd ", dec_states.shape)
            #print (trg_h.shape)
            dec_out = trg_h.squeeze(1)
            #print ("dec_out ",dec_out.shape) #beam_size
            #out = F.softmax(self.decoder.fc(dec_out)).unsqueeze(0)
            out = F.softmax(dec_out).unsqueeze(0)
            #out = dec_out
            word_lk = out.view(#5,1,1002 #probabilities
                beam_size,
                remaining_sents,
                -1
            ).transpose(0, 1).contiguous()#1,5,1002
            #print("word ",word_lk,word_lk.shape)
            active = []
            for b in range(batch_size):
                if beam[b].done:
                    continue

                idx = batch_idx[b]
                if not beam[b].advance(word_lk.data[idx]):
                    active += [b]

                '''
                for dec_state in dec_states:  # iterate over h, c
                    # layers x beam*sent x dim
                    print(dec_state.shape)#5,24
                    size2 = dec_state.size(2)
                    aux = dec_state.view(-1, beam_size, remaining_sents, size2) #remaining_sents =batchsize
                    print(aux.shape)
                    sent_states = aux[:, :, idx]
                    sent_states.data.copy_(
                        sent_states.data.index_select(1,beam[b].get_current_origin())
                    )
                '''
            if not active:
                break

            # in this section, the sentences that are still active are
            # compacted so that the decoder is not run on completed sentences
            if use_cuda:
                active_idx = t.cuda.LongTensor([batch_idx[k] for k in active])
            else:
                active_idx = t.LongTensor([batch_idx[k] for k in active])
            batch_idx = {beam: idx for idx, beam in enumerate(active)}

            def update_active(t):
                # select only the remaining active sentences
                view = t.data.view(
                    -1, remaining_sents,
                    self.params.decoder_rnn_size
                )
                new_size = list(t.size())
                new_size[-2] = new_size[-2] * len(active_idx) \
                    // remaining_sents
                return Variable(view.index_select(
                    1, active_idx
                ).view(*new_size))
            '''
            dec_states = (
                update_active(dec_states[0]),
                #update_active(dec_states[1])
            )
            '''
            dec_states = update_active(dec_states[0])
            #dec_out = update_active(dec_out)
            # context = update_active(context)

            remaining_sents = len(active) 

         # (4) package everything up

        allHyp, allScores = [], []


        for b in range(batch_size):
            scores, ks = beam[b].sort_best()
            # print scores
            # print ks 
            allScores += [scores[:n_best]]
            hyps = zip(*[beam[b].get_hyp(k) for k in ks[:n_best]])
            # print hyps
            # print "------------------"
            allHyp += [hyps]

        # print '==== Complete ========='

        return allHyp, allScores 