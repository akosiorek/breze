import numpy as np
import theano
import theano.tensor as T
from theano.gradient import disconnected_grad
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
    min_diag_var = 0.01

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
        if not self.fixed_var:
            print 'hypervar_mean = 0'
            # hypervar_mean = T.ones((1, n_samples, n_var_par))
            hypervar_mean = T.zeros((1, n_samples, n_var_par))
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
            var = sample[:, :, -n_var_par:] ** 2 + self.min_diag_var
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

    def template(self):
        parts =  '{n_iter:12} {time:12.2f} {loss:12.2f} {val_loss:12.2f} means={means:} vars={vars}'.split()
        if self.annealing:
            parts.append('{beta:12}')
        return ' '.join(parts)

    def report(self, info):
        mean, var = self.hyperparam_model.raw_mean, self.hyperparam_model.raw_var
        mean, var = self.parameters[mean], self.parameters[var]**2
        info['means'] = '; '.join(('%.4f' % s for s in (mean.min(), mean.mean(), mean.max())))
        info['vars'] = '; '.join(('%.4f' % s for s in (var.min(), var.mean(), var.max())))
        if self.annealing:
            info['beta'] = self.current_beta


class PmpRnn(StochasticRnn):
    annealing = False
    beta0 = 0.01
    beta_T = 10000

    def __init__(self, n_inpt, n_hiddens_recog, n_latent, n_hiddens_gen,
                 recog_transfers, gen_transfers,
                 p_dropout_inpt=.1, p_dropout_hiddens=.1,
                 p_dropout_hidden_to_out=None,
                 p_dropout_shortcut=None,
                 use_imp_weight=False,
                 batch_size=None, optimizer='rprop',
                 max_iter=1000, verbose=False, n_samples=None):
        
        self.n_samples = n_samples
        StochasticRnn.__init__(self, n_inpt, n_hiddens_recog, n_latent, n_hiddens_gen,
                 recog_transfers, gen_transfers,
                 p_dropout_inpt, p_dropout_hiddens,
                 p_dropout_hidden_to_out,
                 p_dropout_shortcut,
                 use_imp_weight,
                 batch_size, optimizer,
                 max_iter, verbose)

    def init_train(self, X, VX, TX):

        if hasattr(self, 'n_samples_param'):
            self.parameters[self.n_samples_param] = float(X.shape[1])

        return X, VX, TX

    def anneal(self, n_iter):
        if self.annealing:
            old_val = self.alpha.eval()
            self.iter.set_value(np.asarray([n_iter], dtype=theano.config.floatX))
            return old_val, self.alpha.eval()

    def schedule(self, info):
        if self.annealing:
            self.current_beta = np.minimum(1.0, self.beta0 + float(info['n_iter']) / self.beta_T)
            self.parameters[self.beta] = self.current_beta

        try:
            self._schedule_at_iter(int(info['n_iter']), info)
        except AttributeError:
            pass

    def _init_exprs(self):
        inpt, self.imp_weight = self._make_start_exprs()
        self.parameters = ParameterSet()

        self.beta = self.parameters.declare((1,))
        n_dim = inpt.ndim

        if not self.n_samples:
            self.n_samples_param = self.parameters.declare(1)
            self.n_samples_expr = disconnected_grad(self.n_samples_param).squeeze()
        else:
            self.n_samples_expr = self.n_samples

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
            imp_weight = 1

        rec_loss = self.vae.gen.nll(inpt)
        self.rec_loss_sample_wise = rec_loss.sum(axis=-1)
        self.rec_loss = self.rec_loss_sample_wise.mean()

        output = self.vae.gen.stt
        beta = disconnected_grad(self.beta.mean())

        # Create the KL divergence part of the loss.
        self.kl_coord_wise = imp_weight * kl_div(self.vae.recog, self.vae.prior, beta)
        self.kl_sample_wise = self.kl_coord_wise.sum(axis=-1)
        # self.kl_sample_wise = self._kl_expectation(self.kl_coord_wise.sum(axis=n_dim - 1))
        self.kl = self.kl_sample_wise.mean()

        loss = self.kl
        true_loss = (imp_weight * kl_div(self.vae.recog, self.vae.prior)).sum(axis=-1).mean()

        # Create the KL divergence between model and prior for hyperparams.
        # It is the same for every timestep, so take once instead
        #  of averaging
        try:
            self.hyper_kl_coord_wise = imp_weight * kl_div(self.hyperparam_model, self.hyperprior, beta)[0, :, :]
            self.hyper_kl_weight = 1. / self.n_samples_expr / inpt.shape[0]
            self.hyper_kl_sample_wise = self.hyper_kl_coord_wise.sum(axis=-1)
            self.hyper_kl = self.hyper_kl_sample_wise.mean()
            loss += self.hyper_kl_weight * self.hyper_kl
            true_loss += self.hyper_kl_weight * kl_div(self.hyperparam_model, self.hyperprior, beta)[0, :, :].sum(axis=-1).mean()

        except AttributeError as err:
            print err.message, 'Skipping Hyperprior-related loss.'

        true_loss += self.rec_loss
        loss += disconnected_grad(self.beta.mean()) * self.rec_loss

        UnsupervisedModel.__init__(self, inpt=inpt,
                                   output=output,
                                   loss=loss,
                                   parameters=self.parameters,
                                   imp_weight=self.imp_weight)

        self.transform_expr_name = None
        self.exprs['true_loss'] = true_loss

    def initialize(self,
                   par_std=1, par_std_affine=None, par_std_rec=None,
                   par_std_in=None,
                   sparsify_affine=None, sparsify_rec=None,
                   spectral_radius=None):

        super(PmpRnn, self).initialize(par_std, par_std_affine, par_std_rec,
                                       par_std_in, sparsify_affine,
                                       sparsify_rec, spectral_radius)

        # try:
        #     hyperparam = self.hyperparam_model
        #     self.parameters[hyperparam.raw_mean][:-self.n_latent] = 0
        #     if not hasattr(self, 'fixed_var') or not self.fixed_var:
        #         self.parameters[hyperparam.raw_mean][-self.n_latent:] = 1
        #     self.parameters[hyperparam.raw_var] = 1
        # except AttributeError as err:
        #     print err.message, 'Skipping init.'

        try:
            self.vae.prior.init()
        except AttributeError as err:
            print err.message, 'Prior doesn\'t need init'

        self.parameters[self.beta] = self.beta0
        self.current_beta = self.beta0
