import math
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import torch.nn.functional as F
#from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt 


"""implementation of the Variational Recurrent
Neural Network (VRNN) from https://arxiv.org/abs/1506.02216
using unimodal isotropic gaussian distributions for 
inference, prior, and generating models."""


class VRNN(nn.Module):
	def __init__(self, x_dim, h_dim, z_dim, n_layers,  word_vocab_size, bias=False, use_cuda=False):
		super(VRNN, self).__init__()

		self.use_cuda = use_cuda
		self.x_dim = x_dim
		self.h_dim = h_dim
		self.z_dim = z_dim
		self.n_layers = n_layers
		self.word_vocab_size = word_vocab_size

		#feature-extracting transformations
		self.phi_x = nn.Sequential(
			nn.Linear(x_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, h_dim),
			nn.ReLU())
		self.phi_z = nn.Sequential(
			nn.Linear(z_dim, h_dim),
			nn.ReLU())

		#encoder
		self.enc = nn.Sequential(
			nn.Linear(h_dim + h_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, h_dim),
			nn.ReLU())
		self.enc_mean = nn.Linear(h_dim, z_dim)
		self.enc_std = nn.Sequential(
			nn.Linear(h_dim, z_dim),
			nn.Softplus())

		#prior
		self.prior = nn.Sequential(
			nn.Linear(h_dim, h_dim),
			nn.ReLU())
		self.prior_mean = nn.Linear(h_dim, z_dim)
		self.prior_std = nn.Sequential(
			nn.Linear(h_dim, z_dim),
			nn.Softplus())

		#decoder
		self.dec = nn.Sequential(
			nn.Linear(h_dim + h_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, h_dim),
			nn.ReLU())
		self.dec_std = nn.Sequential(
			nn.Linear(h_dim, x_dim),
			nn.Softplus())
		#self.dec_mean = nn.Linear(h_dim, x_dim)
		self.dec_mean = nn.Sequential(
			nn.Linear(h_dim, x_dim),
			nn.Sigmoid())

		#recurrence
		self.rnn = nn.GRU(h_dim + h_dim, h_dim, n_layers, bias)
		self.fc = nn.Linear(x_dim, self.word_vocab_size)

	def only_decoder_beam(self, decoder_input, z, drop_prob, initial_state=None):
        
		#assert parameters_allocation_check(self), \
		#    'Invalid CUDA options. Parameters should be allocated in the same memory'

		#         print decoder_input.size()
		 
		[beam_batch_size, _, _] = decoder_input.size()

		'''
		    decoder rnn is conditioned on context via additional bias = W_cond * z to every input token
		'''
		decoder_input = F.dropout(decoder_input, drop_prob)

		z = z.unsqueeze(0)

		#         print z.size()

		z = torch.cat([z] * beam_batch_size, 0)

		#         print z.size()
		#         z = z.contiguous().view(1, -1)

		#         z = z.view(beam_batch_size, self.params.latent_variable_size)

		#         print z.size() 

		decoder_input = torch.cat([decoder_input, z], 2)

		#         print "decoder_input:",decoder_input.size() 

		rnn_out, final_state = self.rnn(decoder_input, initial_state) 

		#         print "rnn_out:",rnn_out.size()
		#         print "final_state_1:",final_state[0].size()
		#         print "final_state_1:",final_state[1].size()   

		return rnn_out, final_state

	def forward(self, x, z_global, drop_prob, initial_state, sampling=False): #decoder_input, z, drop_prob, initial_state
		'''
        :x is expected as [t,b,v] param decoder_input: tensor with shape of [batch_size, seq_len, embed_size]
        :param z: sequence context with shape of [batch_size, latent_variable_size]
        :param drop_prob: probability of an element of decoder input to be zeroed in sense of dropout
        :param initial_state: initial state of decoder rnn
		'''
		#print(len(initial_state))
		all_enc_mean, all_enc_std = [], []
		all_dec_mean, all_dec_std = [], []
		outputs = []
		kld_loss = 0
		nll_loss = 0
		x = F.dropout(x, drop_prob)
		x = x.transpose(0, 1) #[seq_len, batch_size, vector_dim]
		#print( "tipo init ",type(initial_state[0]))
		
		''' RVAE decoder
		z = t.cat([z] * seq_len, 1).view(batch_size, seq_len, self.params.latent_variable_size)
        decoder_input = t.cat([decoder_input, z], 2)

        rnn_out, final_state = self.rnn(decoder_input, initial_state)

        rnn_out = rnn_out.contiguous().view(-1, self.params.decoder_rnn_size)
        result = self.fc(rnn_out)
        result = result.view(batch_size, seq_len, self.params.word_vocab_size)
        '''
        #*2 because of bidirectional component
		h = initial_state.view(self.n_layers, x.size(1), self.h_dim)# Variable(torch.zeros(self.n_layers, x.size(1), self.h_dim))
		if self.use_cuda:
			h = h.cuda() 
		for t in range(x.size(0)):
			
			phi_x_t = self.phi_x(x[t])
			#encoder
			enc_t = self.enc(torch.cat([phi_x_t, h[-1]], 1))
			enc_mean_t = self.enc_mean(enc_t)
			enc_std_t = self.enc_std(enc_t)

			#prior
			prior_t = self.prior(h[-1])
			prior_mean_t = self.prior_mean(prior_t)
			prior_std_t = self.prior_std(prior_t)

			#sampling and reparameterization
			if sampling:
				z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)
			else:
				z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
			phi_z_t = self.phi_z(z_t)

			#decoder
			dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
			dec_mean_t = self.dec_mean(dec_t)
			dec_std_t = self.dec_std(dec_t)

			eps = Variable(torch.randn(dec_std_t.size(0), self.x_dim))
			if self.use_cuda:
				eps = eps.cuda()
			aux = torch.exp(dec_std_t / 2) * eps
			pred_we = dec_mean_t + aux
			outputs.append(pred_we)

			#recurrence
			_, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

			#computing losses
			kld_loss += self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
			#nll_loss += self._nll_gauss(dec_mean_t, dec_std_t, x[t])
			nll_loss += self._nll_bernoulli(dec_mean_t, x[t])

			#outputs.append(output) #rvae
			all_enc_std.append(enc_std_t.detach())
			all_enc_mean.append(enc_mean_t.detach())
			all_dec_mean.append(dec_mean_t.detach())
			all_dec_std.append(dec_std_t.detach())

		outputs = torch.stack(outputs)
		rnn_out = outputs.view(-1, self.x_dim)
		result = self.fc(rnn_out)
		result = result.view(-1, x.size(0), self.word_vocab_size) #batch_size, seq_len, self.word_vocab_size

		return kld_loss, nll_loss, \
			(all_enc_mean, all_enc_std), \
			(all_dec_mean, all_dec_std), \
			result, h



	def sample(self, seq_len):

		sample = torch.zeros(seq_len, self.x_dim)

		h = Variable(torch.zeros(self.n_layers, 1, self.h_dim))
		if self.use_cuda:
			h = h.cuda()
		for t in range(seq_len):

			#prior
			prior_t = self.prior(h[-1])
			prior_mean_t = self.prior_mean(prior_t)
			prior_std_t = self.prior_std(prior_t)

			#sampling and reparameterization
			z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)
			phi_z_t = self.phi_z(z_t)
			
			#decoder
			dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
			dec_mean_t = self.dec_mean(dec_t)
			#dec_std_t = self.dec_std(dec_t)

			phi_x_t = self.phi_x(dec_mean_t)

			#recurrence
			_, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

			sample[t] = dec_mean_t.data
	
		return sample


	def reset_parameters(self, stdv=1e-1):
		for weight in self.parameters():
			weight.data.normal_(0, stdv)


	def _init_weights(self, stdv):
		pass


	def _reparameterized_sample(self, mean, std):
		"""using std to sample"""
		if self.use_cuda:
			eps = torch.cuda.FloatTensor(std.size()).normal_()
		else:
			eps = torch.FloatTensor(std.size()).normal_()
		eps = Variable(eps)
		return eps.mul(std).add_(mean)


	def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
		"""Using std to compute KLD"""

		kld_element =  (2 * torch.log(std_2) - 2 * torch.log(std_1) + 
			(std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
			std_2.pow(2) - 1)
		return	0.5 * torch.sum(kld_element)


	def _nll_bernoulli(self, theta, x):
		return - torch.sum(x*torch.log(theta) + (1-x)*torch.log(1-theta))


	def _nll_gauss(self, mean, std, x):
		pass