import argparse
import os

import numpy as np
import torch as t
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, MultiStepLR

from utils.batch_loader import BatchLoader
from utils.parameters import Parameters
from model.rvae import RVAE



if __name__ == "__main__":

    if not os.path.exists('data/word_embeddings.npy'):
        raise FileNotFoundError("word embeddings file was't found")

    parser = argparse.ArgumentParser(description='RVAE')
    parser.add_argument('--num-iterations', type=int, default=1000, metavar='NI', #120000
                        help='num iterations (default: 120000)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='BS',
                        help='batch size (default: 32)')
    parser.add_argument('--use-VRNN', action='store_true',#default value True #type=bool, default=True, metavar='CUDA',
                        help='use VRNN as decoder (default: False)')
    parser.add_argument('--use-cuda', action='store_true',#
                        help='use cuda (default: False)')
    parser.add_argument('--learning-rate', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.00005)')
    parser.add_argument('--dropout', type=float, default=0.3, metavar='DR',
                        help='dropout (default: 0.3)')
    parser.add_argument('--use-trained', type=bool, default=False, metavar='UT',
                        help='load pretrained model (default: False)')
    parser.add_argument('--ce-result', default='', metavar='CE',
                        help='ce result path (default: '')')
    parser.add_argument('--kld-result', default='', metavar='KLD',
                        help='ce result path (default: '')')

    args = parser.parse_args()

    batch_loader = BatchLoader('')
    parameters = Parameters(batch_loader.max_word_len,
                            batch_loader.max_seq_len,
                            batch_loader.words_vocab_size,
                            batch_loader.chars_vocab_size)

    rvae = RVAE(parameters, args.use_VRNN)
    if args.use_trained:
        rvae.load_state_dict(t.load('trained_RVAE'))
    if args.use_cuda:
        print("Using cuda")
        rvae = rvae.cuda()

    optimizer = Adam(rvae.learnable_parameters(), args.learning_rate)

    train_step = rvae.trainer(optimizer, batch_loader)
    validate = rvae.validater(batch_loader)

    ce_result = []
    kld_result = []

    scheduler = MultiStepLR(optimizer, milestones=[100], gamma=0.1)
    for iteration in range(args.num_iterations):
        scheduler.step()
        cross_entropy, kld, coef,loss = train_step(iteration, args.batch_size, args.use_cuda, args.dropout)

        '''
        if iteration % 5 == 0:
            print('\n')
            print('------------TRAIN-------------')
            print('Iteration,cross_entropy,kld,coef')
            print(iteration,cross_entropy.data.cpu().numpy(),kld.data.cpu().numpy(),coef)
            print('------------------------------')
        '''
        print('Train-it: {}, loss: {:.4f}, cross_entropy: {:.4f}, KL: {:.4f}, coef {:8f}'.format(iteration, loss, cross_entropy/args.batch_size, kld/args.batch_size,coef))
        '''
        if iteration % 10 == 0:
            instance = 5
            cross_entropyVal, kldVal, softmax, lossVal , inputEnco= validate(iteration,args.batch_size, args.use_cuda, instance)

            cross_entropyVal = cross_entropyVal.data.cpu().numpy()#[0]
            kldVal = kldVal.data.cpu().numpy()#[0]
            softmax = softmax.data.cpu().numpy()

            resultInput = ''
            for idWord in inputEnco:
                word = batch_loader.idx_to_word[idWord]
                #word2= batch_loader.sample_word_from_distribution(softmax.data.cpu().numpy()[-1])
                resultInput += ' ' + word

            ids = np.argmax(softmax, axis=1)
            # added by Gissella to see the validation words generated
            #lenSize = len(softmax)/args.batch_size
            ids = np.reshape(ids,[args.batch_size, -1])

            result = ''
            for idWord in ids[instance]:
                word = batch_loader.idx_to_word[idWord]
                #word2= batch_loader.sample_word_from_distribution(softmax.data.cpu().numpy()[-1])
                result += ' ' + word
                #result2 += ' ' + word2
            print("------- VALIDATION EXAMPLE ---------")
            print("Real: ",resultInput)
            print("Vali: ", result)
            print('\n')



            print('Validation-it: {}, loss-val: {:.4f}, cross_entropy: {:.4f}, KL: {:.4f}, coef {:8f}'.format(iteration, lossVal, cross_entropyVal/args.batch_size, kldVal/args.batch_size,coef))
        '''
        if iteration % 10 == 0:
            ce_result += [cross_entropy]
            kld_result += [kld]

            seed = np.random.normal(size=[1, parameters.latent_variable_size])
            sample = rvae.sample(batch_loader, 50, seed, args.use_cuda)
            print('------------SAMPLE------------')
            print(sample)

    t.save(rvae.state_dict(), 'trained_RVAE')

    np.save('output/ce_result_{}.npy'.format(args.ce_result), np.array(ce_result.data.cpu().numpy()))
    np.save('output/kld_result_npy_{}'.format(args.kld_result), np.array(kld_result.data.cpu().numpy()))
