import numpy as np

def fold(f, l, a):
    return a if (len(l) == 0) else fold(f, l[1:], f(a, l[0]))


def f_and(x, y):
    return x and y


def f_or(x, y):
    return x or y


def parameters_allocation_check(module):
    parameters = list(module.parameters())
    return fold(f_and, parameters, True) or not fold(f_or, parameters, False)


def handle_inputs(inputs, use_cuda):
    import torch as t
    from torch.autograd import Variable

    result = [Variable(t.from_numpy(var)) for var in inputs]
    result = [var.cuda() if use_cuda else var for var in result]

    return result


def kld_coef(i):
    import math
    k=0.025
    x0=100
    coef = float(1/(1+np.exp(-k*(i-x0))))
    return coef#(math.tanh((i - 500)/500) + 1)/2

def kl_anneal_function(step, totalIt, anneal_function='logistic'):

    k=0.0025
    x0=2500
    if anneal_function == 'logistic':
        return float(1/(1+np.exp(-k*(step-x0))))
    elif anneal_function == 'linear':
        x0 = totalIt
        return min(1, step/x0)