# -*- coding: utf-8 -*-

import numpy as np
import theano.tensor as T

from distributions import DiagGauss, NormalGauss, Bernoulli, NormalizingFlow

from breze.arch.component.misc import inter_gauss_kl

from breze.arch.util import wild_reshape

def recover_time(X, time_steps):
    return wild_reshape(X, (time_steps, -1, X.shape[1]))

def normflow_gauss_kl(p, q):
    #kl = - p.initial_dist.entropy()
    kl = - p.initial_dist.nll(p.z_0)
    kl += q.nll(p.z)
    # since kl is coord wise and taking the sum later on
    # would multiply it with z.shape[1]
    if p.z.ndim == 3:
        kl += wild_reshape(p.neglogdet, (p.z.shape[0], -1)).dimshuffle(0, 1,
                                                                     'x') / p.z.shape[2]
    else:
        kl += p.neglogdet.dimshuffle(0, 'x') / p.z.shape[1]
    return kl


def gauss_normalgauss_kl(p, q, beta=None):
    kl = inter_gauss_kl(p.mean, p.var, beta=beta, var_offset=1e-4)
    return kl


def gauss_gauss_kl(p, q, beta=None):
    kl = inter_gauss_kl(p.mean, p.var, q.mean, q.var, beta=beta)
    return kl


def bern_bern_kl(p, q, beta=None):
    p_rate = p.rate
    p_rate *= 0.999
    p_rate += 0.0005

    q_rate = q.rate
    q_rate *= 0.999
    q_rate += 0.0005

    return (p_rate * T.log(p_rate / q_rate) + \
           (1 - p_rate) * T.log((1 - p_rate)/(1 - q_rate)))


kl_table = {
    (DiagGauss, NormalGauss): gauss_normalgauss_kl,
    (DiagGauss, DiagGauss): gauss_gauss_kl,
    (Bernoulli,  Bernoulli): bern_bern_kl,
    (NormalizingFlow, NormalGauss): normflow_gauss_kl,
    (NormalizingFlow, DiagGauss): normflow_gauss_kl
}


def kl_div(p, q, beta=None, sample=False):
    if not sample:
        for i in kl_table:
            if isinstance(p, i[0]) and isinstance(q, i[1]):
                return kl_table[i](p, q, beta=beta)
        raise NotImplementedError(
            'unknown distribution combo: %s and %s' % (type(p), type(q)))
    else:
        # TODO: implement kl through sampling
        raise NotImplemented()
