import numpy as np
import theano
import theano.tensor as T
import climin.initialize

from breze.arch.util import ParameterSet
from breze.arch.construct.layer.distributions import NormalGauss, RankOneGauss
from breze.arch.construct.layer.kldivergence import kl_div
from breze.arch.construct.sgvb import (
    VariationalAutoEncoder as _VariationalAutoEncoder)
from breze.arch.construct.neural.base import wild_reshape
from breze.learn.base import UnsupervisedModel

from base import GenericVariationalAutoEncoder
from prior import LearnableDiagGauss
from storn import StochasticRnn


class ProbabilisticMovementPrimitive(RankOneGauss):

    def __init__(self, n_basis, mean, var, u, rng=None):

        self.n_basis = n_basis
        mean, u = (self._transform(i) for i in (mean, u))
        super(ProbabilisticMovementPrimitive, self).__init__(mean, var, u, rng)

    def _transform(self, x):

        n_time_steps, n_samples, n_dims = x.shape
        x = wild_reshape(x, (n_time_steps, n_samples, self.n_basis, -1))

        indices = T.constant(np.linspace(0, 1, self.n_basis), dtype=theano.config.floatX)
        timesteps = T.arange(0, 1, 1. / n_time_steps)

        def times_basis(tens, t, b):
            basis = T.exp(-(t - b) ** 2 / 2)
            return T.dot(basis / basis.sum(), tens)

        x, _ = theano.scan(times_basis, sequences=[x, timesteps], non_sequences=indices)
        return x


# hyper-prior - NormalGauss
# parametric prior - ProbabilisticMovementPrimitive
# distrib over primitive weights - LearnableDiagGauss
# recognition model - stochastic RNN
# generating model - stochastic RNN


class PmpPriorMixin(object):
    def make_prior(self, recog_sample, recog=None):

        n_timesteps, n_samples, _ = recog_sample.shape
        n_mean_par = self.n_latent * self.n_bases
        n_dims = n_mean_par + self.n_latent
        shape = (n_timesteps, n_samples, n_dims)

        self.hyperprior = NormalGauss(shape)
        self.hyperparam_model = LearnableDiagGauss(shape, n_dims,
                                                   self.parameters.declare)

        sample = self.hyperparam_model.sample()
        mean = sample[:, :, :n_mean_par]
        var = sample[:, :, n_mean_par:] ** 2
        u = mean

        return ProbabilisticMovementPrimitive(self.n_bases, mean, var, u)


class PmpRnn(StochasticRnn):

    def _init_exprs(self):
        inpt, self.imp_weight = self._make_start_exprs()
        self.parameters = ParameterSet()

        n_dim = inpt.ndim

        self.vae = _VariationalAutoEncoder(inpt, self.n_inpt,
                                           self.n_latent, self.n_output,
                                           self.make_recog,
                                           self.make_prior,
                                           self.make_gen,
                                           getattr(self, 'make_cond', None),
                                           declare=self.parameters.declare)

        self.recog_sample = self.vae.recog_sample

        if self.use_imp_weight:
            imp_weight = T.addbroadcast(self.imp_weight, n_dim - 1)
        else:
            imp_weight = False

        rec_loss = self.vae.gen.nll(inpt)
        self.rec_loss_sample_wise = rec_loss.sum(axis=n_dim - 1)
        self.rec_loss = self.rec_loss_sample_wise.mean()

        output = self.vae.gen.stt

        # Create the KL divergence part of the loss.
        n_dim = inpt.ndim
        self.kl_coord_wise = kl_div(self.vae.recog, self.vae.prior)

        if self.use_imp_weight:
            self.kl_coord_wise *= imp_weight
        self.kl_sample_wise = self.kl_coord_wise.sum(axis=n_dim - 1)
        self.kl = self.kl_sample_wise.mean()

        # Create the KL divergence between model and prior for hyperparams.
        self.hyper_kl_coord_wise = kl_div(self.hyperparam_model, self.hyperprior)
        if self.use_imp_weight:
            self.hyper_kl_coord_wise *= imp_weight
        self.hyper_kl_sample_wise = self.hyper_kl_coord_wise.sum(axis=n_dim - 1)
        self.hyper_kl = self.hyper_kl_sample_wise.mean()


        # TODO: scale hyper_kl
        loss = self.kl + self.rec_loss + self.hyper_kl

        UnsupervisedModel.__init__(self, inpt=inpt,
                                   output=output,
                                   loss=loss,
                                   parameters=self.parameters,
                                   imp_weight=self.imp_weight)

        self.transform_expr_name = None

    def initialize(self,
                   par_std=1, par_std_affine=None, par_std_rec=None,
                   par_std_in=None,
                   sparsify_affine=None, sparsify_rec=None,
                   spectral_radius=None):

        super(PmpRnn, self).initialize(par_std, par_std_affine, par_std_rec,
                                       par_std_in, sparsify_affine,
                                       sparsify_rec, spectral_radius)

        #   TODO: further inits