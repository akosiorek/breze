# -*- coding: utf-8 -*-


import climin.initialize
import numpy as np
import theano
import theano.tensor as T
import theano.tensor.nnet

from theano.compile import optdb

from breze.arch.construct.base import Layer
from breze.arch.construct.layer.distributions import NormalGauss, DiagGauss, RankOneGauss
from breze.arch.construct.layer.varprop.sequential import FDRecurrent
from breze.arch.construct.layer.varprop.simple import AffineNonlinear
from breze.arch.construct.neural import distributions as neural_dists
from breze.learn.utils import theano_floatx
from breze.arch.construct.neural.base import wild_reshape
from breze.arch.construct.layer.varprop import simple as vp_simple

from base import GenericVariationalAutoEncoder
from storn import LatentPriorNormalGaussMixin, GaussLatentStornMixin, ConstVarGaussVisibleStornMixin


class LatentPriorNormalGaussMixin(object):

    def make_prior(self, sample, recog=None):
        return NormalGauss(sample.shape)


class LearnableDiagGauss(Layer, DiagGauss):

    def __init__(self, shape, n_features, declare=None):

        self.shape = shape
        self.n_features = n_features
        Layer.__init__(self, declare=declare)

    def _forward(self):
        n = (self.n_features,)
        self.raw_mean = self.declare(n)
        self.raw_var = self.declare(n)

        ndim = len(self.shape)
        time_steps, batch_size, _ = self.shape
        shape = (time_steps, batch_size, 1)

        mean, var = (T.tile(i, shape, ndim=ndim) for i in (self.raw_mean, self.raw_var))
        DiagGauss.__init__(self, mean, var ** 2)


class LearnableRankOneGauss(Layer, RankOneGauss):

    def __init__(self, inpt, n_features, declare=None):

        self.inpt = inpt
        self.n_features = n_features
        Layer.__init__(self, declare=declare)

    def _forward(self):
        n = (self.n_features,)
        self.raw_mean = self.declare(n)
        self.raw_var = self.declare(n)
        self.raw_u = self.declare(n)

        ndim = self.inpt.ndim
        time_steps, batch_size, _ = self.inpt.shape
        shape = (time_steps, batch_size, 1)

        mean, var, u = (T.tile(i, shape, ndim=ndim) for i in
                        (self.raw_mean, self.raw_var, self.raw_u))

        RankOneGauss.__init__(self, mean, var ** 2, u)


class DoublyRandomGauss(LearnableDiagGauss):

    def __init__(self, inpt, n_output, declare=None):

        self.n_output = n_output
        super(DoublyRandomGauss, self).__init__(inpt, 3 * n_output, declare)

    def _forward(self):
        super(DoublyRandomGauss, self)._forward()

        sample = super(DoublyRandomGauss, self).sample()
        mean = sample[:, :, :self.n_output]
        var = sample[:, :, self.n_output:2 * self.n_output]
        u = sample[:, :, 2 * self.n_output:]
        self.output_distrib = RankOneGauss(mean, var ** 2, u)
        # self.output_distrib = DiagGauss(mean, var**2)

    def sample(self, epsilon=None):
        return self.output_distrib.sample(epsilon)


class DoublyRandomGaussRecogMixin(object):

    def make_recog(self, inpt):
        return DoublyRandomGauss(inpt, self.n_latent, self.parameters.declare)


class RankOneGaussRecogMixin(object):
    def make_recog(self, inpt):
        return LearnableRankOneGauss(inpt, self.n_latent, self.parameters.declare)


class DoublyStochasticPrior(GenericVariationalAutoEncoder):

    sample_dim = 1,
    theano_optimizer = optdb.query(theano.gof.Query(
        include=['fast_run'], exclude=['scan_eqopt1', 'scan_eqopt2']))
    mode = theano.Mode(linker='cvm', optimizer=theano_optimizer)

    def __init__(self, n_inpt, n_latent, n_hiddens_gen,
                 gen_transfers,
                 p_dropout_inpt=.1, p_dropout_hiddens=.1,
                 p_dropout_hidden_to_out=None,
                 p_dropout_shortcut=None,
                 use_imp_weight=False,
                 batch_size=None, optimizer='rprop',
                 max_iter=1000, verbose=False):

        self.n_inpt = n_inpt
        self.n_hiddens_gen = n_hiddens_gen
        self.gen_transfers = gen_transfers

        self.p_dropout_inpt = p_dropout_inpt
        self.p_dropout_hiddens = p_dropout_hiddens
        self.p_dropout_hidden_to_out = p_dropout_hidden_to_out
        self.p_dropout_shortcut = p_dropout_shortcut

        super(DoublyStochasticPrior, self).__init__(
            n_inpt, n_latent,
            use_imp_weight=use_imp_weight,
            batch_size=batch_size, optimizer=optimizer,
            max_iter=verbose, verbose=verbose)

    def _make_start_exprs(self):
        inpt = T.tensor3('inpt')
        inpt.tag.test_value, = theano_floatx(np.ones((4, 3, self.n_inpt)))

        if self.use_imp_weight:
            imp_weight = T.tensor3('imp_weight')
            imp_weight.tag.test_value, = theano_floatx(np.ones((4, 3, 1)))
        else:
            imp_weight = None

        return inpt, imp_weight

    def make_cond(self, inpt):
        return T.concatenate([T.zeros_like(inpt[:1]), inpt[:-1]], 0)

    def sample(self):
        self.vae.recog_sample


class Prior(DoublyStochasticPrior,
            LatentPriorNormalGaussMixin,
            # DoublyRandomGaussRecogMixin,
            RankOneGaussRecogMixin,
            ConstVarGaussVisibleStornMixin):
    shared_std = True

# class Storn(StochasticRnn,
#             GaussLatentStornMixin,
#             ConstVarGaussVisibleStornMixin):
#     shared_std = True




