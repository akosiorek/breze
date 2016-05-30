import numpy as np
import theano
import theano.tensor as T
import climin.initialize

from breze.arch.util import ParameterSet
from breze.arch.construct.layer.distributions import NormalGauss, RankOneGauss, DiagGauss
from breze.arch.construct.layer.kldivergence import kl_div
from breze.arch.construct.sgvb import (
    VariationalAutoEncoder as _VariationalAutoEncoder)
from breze.arch.construct.neural.base import wild_reshape
from breze.learn.base import UnsupervisedModel, cast_array_to_local_type

from base import GenericVariationalAutoEncoder
from prior import LearnableDiagGauss
from storn import StochasticRnn


class ProbabilisticMovementPrimitive(RankOneGauss):

    def __init__(self, n_basis, mean, var, u, rng=None, width=1, eps=None, parameters=None):

        self.parameters = parameters
        self.n_basis = n_basis

        self.width = self._width() if hasattr(self, '_width') else width
        mean, u = (self._transform(i) for i in (mean, u))

        if eps is not None:
            super(ProbabilisticMovementPrimitive, self).__init__(mean, var, u, rng, eps)
        else:
            super(ProbabilisticMovementPrimitive, self).__init__(mean, var, u, rng)

    def _transform(self, x):

        n_time_steps, n_samples, n_dims = x.shape
        x = wild_reshape(x, (n_time_steps, n_samples, self.n_basis, -1))

        self.indices = self._indices()
        timesteps = T.arange(0, 1, 1. / n_time_steps)

        def times_basis(tens, t, b, w):
            basis = T.exp(-(t - b) ** 2 / (2 * w))
            return T.dot(basis / basis.sum(), tens)

        x, _ = theano.scan(times_basis, sequences=[x, timesteps], non_sequences=[self.indices, self.width])
        return x

    def _indices(self):
        return T.constant(np.linspace(0, 1, self.n_basis), 'basis_time_indices', dtype=theano.config.floatX)

    def init(self):
        pass


class LegendreProbabilisticMovementPrimitive(ProbabilisticMovementPrimitive):

    def _indices(self):
        from gaussian_roots import roots
        return T.constant(0.5 * np.asarray(roots[self.n_basis]) + 0.5, 'basis_time_indices', dtype=theano.config.floatX)


class FullyLearnablePMP(ProbabilisticMovementPrimitive):

    def _width(self):
        return self.parameters.declare(self.n_basis)
        # return self.parameters.declare(1)

    def _indices(self):
        return self.parameters.declare(self.n_basis)

    def init(self):
        self.parameters[self.width] = 0.2
        self.parameters[self.indices] = np.linspace(0, 1, self.n_basis)



# hyper-prior - NormalGauss
# parametric prior - ProbabilisticMovementPrimitive
# distrib over primitive weights - LearnableDiagGauss
# recognition model - stochastic RNN
# generating model - stochastic RNN

import sys
sys.setrecursionlimit(50000)


class NormalGaussHyperparamMixin(object):
    def _make_hyperparam_model(self, shape):
        return NormalGauss(shape)


class DiagGaussHyperparamMixin(object):
    def _make_hyperparam_model(self, shape):
        return LearnableDiagGauss(shape, shape[-1], self.parameters.declare)


class PmpPriorMixin(object):
    kl_samples = 1
    pmp_width = 1
    pmp_class = ProbabilisticMovementPrimitive
    separate_u_mean = False
    fixed_var = False

    def make_prior(self, recog_sample, recog=None):

        n_timesteps, n_samples, _ = recog_sample.shape
        n_mean_par = self.n_latent * self.n_bases
        n_u_par = n_mean_par if self.separate_u_mean else 0
        n_var_par = self.n_latent if not self.fixed_var else 0
        n_dims = n_mean_par + n_u_par + n_var_par

        # we sample once per timeseries
        shape = (1, n_samples, n_dims)

        #   set hypermean to 0 for mean and to 1 for variance, while keeping hypervar = 1 for all
        hyperprior_var = T.ones(shape)
        hypermean_mean = T.zeros((1, n_samples, n_mean_par + n_u_par))
        # if not self.fixed_var:
        hypervar_mean = T.ones((1, n_samples, n_var_par))
        hyperprior_mean = T.concatenate([hypermean_mean, hypervar_mean], axis=len(shape)-1)

        self.hyperprior = DiagGauss(hyperprior_mean, hyperprior_var)
        self.hyperparam_model = self._make_hyperparam_model(shape)

        rng = getattr(self, 'rng', T.shared_randomstreams.RandomStreams())
        self.noises = [rng.normal(size=shape[1:]) for _ in xrange(self.kl_samples)]
        sample = self.hyperparam_model.sample(epsilon=self.noises[0])

        # make the sample available for all timesteps
        sample = T.tile(sample, (n_timesteps, 1, 1), ndim=len(shape))

        mean = sample[:, :, :n_mean_par]
        if self.separate_u_mean:
            u = sample[:, :, n_mean_par:n_mean_par + n_u_par]
        else:
            u = mean

        if not self.fixed_var:
            var = sample[:, :, -n_var_par:] ** 2
        else:
            var = self.fixed_var * T.ones((1, n_samples, self.n_latent))
            var = T.tile(var, (n_timesteps, 1, 1), ndim=len(shape))

        return self.pmp_class(self.n_bases, mean, var, u, width=self.pmp_width, parameters=self.parameters)

    def _kl_expectation(self, kl_estimate):
        if self.kl_samples == 1:
            return kl_estimate

        n0 = self.noises[0]
        kls = [theano.clone(kl_estimate, {n0: n}) for n in self.noises]
        return sum(kls) / self.kl_samples

    def report(self):
        mean, var = self.hyperparam_model.raw_mean, self.hyperparam_model.raw_var
        mean, var = self.parameters[mean], self.parameters[var]**2
        means = ('%.4f' % s for s in (mean.min(), mean.mean(), mean.max()))
        vars = ('%.4f' % s for s in (var.min(), var.mean(), var.max()))

        return 'hyperior mean: {};\tvar: {}'.format(', '.join(means), ', '.join(vars))


class PmpRnn(StochasticRnn):
    annealing = False
    anneal_iters = 10000

    def anneal(self, n_iter):
        if self.annealing:
            old_val = self.alpha.eval()
            self.iter.set_value(np.asarray([n_iter], dtype=theano.config.floatX))
            return old_val, self.alpha.eval()

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

        # self.kl_sample_wise = self.kl_coord_wise.sum(axis=n_dim - 1)
        self.kl_sample_wise = self._kl_expectation(self.kl_coord_wise.sum(axis=n_dim - 1))

        self.kl = self.kl_sample_wise.mean()

        loss = self.kl
        annealed_loss = self.rec_loss

        # Create the KL divergence between model and prior for hyperparams.
        # It is the same for every timestep, so take once instead
        #  of averaging
        self.alpha = self._make_anneal()
        try:
            self.hyper_kl_coord_wise = kl_div(self.hyperparam_model, self.hyperprior)[0, :, :]
            if self.use_imp_weight:
                self.hyper_kl_coord_wise *= imp_weight

            self.hyper_kl_sample_wise = self.hyper_kl_coord_wise.sum(axis=-1)
            self.hyper_kl = self.hyper_kl_sample_wise.mean()

            # latent_sample = self.vae.recog_sample
            # latent_rec_loss = self.vae.prior.nll(latent_sample)
            # self.latent_rec_loss_sample_wise = latent_rec_loss.sum(axis=n_dim - 1)
            # self.latent_rec_loss = self.latent_rec_loss_sample_wise.mean()

            loss += self.hyper_kl
            # annealed_loss += self.latent_rec_loss

        except AttributeError as err:
            print err.message, 'Skipping Hyperior-related loss.'

        true_loss = loss + annealed_loss
        loss += self.alpha * annealed_loss

        UnsupervisedModel.__init__(self, inpt=inpt,
                                   output=output,
                                   loss=loss,
                                   parameters=self.parameters,
                                   imp_weight=self.imp_weight)

        self.transform_expr_name = None
        self.exprs['true_loss'] = true_loss

    def _make_anneal(self):
        if self.annealing:
            self.iter = theano.shared(np.zeros(1, dtype=theano.config.floatX))
            anneal_rate = 0.01 + self.iter / float(self.anneal_iters)
            arg = T.concatenate([T.ones_like(self.iter), anneal_rate])
            alpha = T.min(arg)
        else:
            alpha = 1
        return alpha

    def initialize(self,
                   par_std=1, par_std_affine=None, par_std_rec=None,
                   par_std_in=None,
                   sparsify_affine=None, sparsify_rec=None,
                   spectral_radius=None):

        super(PmpRnn, self).initialize(par_std, par_std_affine, par_std_rec,
                                       par_std_in, sparsify_affine,
                                       sparsify_rec, spectral_radius)

        try:
            hyperparam = self.hyperparam_model
            self.parameters[hyperparam.raw_mean][:-self.n_latent] = 0
            if not hasattr(self, 'fixed_var') or not self.fixed_var:
                self.parameters[hyperparam.raw_mean][-self.n_latent:] = 1
            self.parameters[hyperparam.raw_var] = 1
        except AttributeError as err:
            print err.message, 'Skipping init.'

        try:
            self.vae.prior.init()
        except AttributeError as err:
            print err.message, 'Prior doesn\'t need init'
