# -*- coding: utf-8 -*-

import theano.tensor as T

from distributions import DiagGauss, NormalGauss, RankOneGauss, Bernoulli

from breze.arch.component.misc import inter_gauss_kl


def gauss_normalgauss_kl(p, q):
    kl = inter_gauss_kl(p.mean, p.var, var_offset=1e-4)
    return kl


def gauss_gauss_kl(p, q):
    kl = inter_gauss_kl(p.mean, p.var, q.mean, q.var)
    return kl


def diaggauss_affinegauss_kl(p, q):
    return inter_gauss_kl(p.mean, p.var, q.mean, q.var, u=q.u, eta=q.eta)



def bern_bern_kl(p, q):
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
    (DiagGauss, RankOneGauss): diaggauss_affinegauss_kl,
    (Bernoulli,  Bernoulli): bern_bern_kl
}


def kl_div(p, q, sample=False):
    if not sample:
        for i in kl_table:
            if isinstance(p, i[0]) and isinstance(q, i[1]):
                return kl_table[i](p, q)
        raise NotImplementedError(
            'unknown distribution combo: %s and %s' % (type(p), type(q)))
    else:
        # TODO: implement kl through sampling
        raise NotImplemented()
