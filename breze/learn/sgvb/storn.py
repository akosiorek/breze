# -*- coding: utf-8 -*-


import climin.initialize
import numpy as np
import theano
import theano.tensor as T
import theano.tensor.nnet

from theano.compile import optdb

from breze.arch.construct.layer.distributions import NormalGauss, DiagGauss, RankOneGauss
from breze.arch.construct.layer.varprop.sequential import FDRecurrent
from breze.arch.construct.layer.varprop.simple import AffineNonlinear
from breze.arch.construct.neural import distributions as neural_dists
from breze.learn.utils import theano_floatx

from base import GenericVariationalAutoEncoder


class BernoulliVisibleStornMixin(object):

    def make_gen(self, latent_sample):
        return neural_dists.FastDropoutRnnBernoulli(
            latent_sample,
            n_inpt=self.n_latent + self.n_inpt,
            n_hiddens=self.n_hiddens_gen,
            n_output=self.n_inpt,
            hidden_transfers=self.gen_transfers,
            p_dropout_inpt=self.p_dropout_inpt,
            p_dropout_hiddens=self.p_dropout_hiddens,
            p_dropout_hidden_to_out=self.p_dropout_hidden_to_out,
            declare=self.parameters.declare)


class GaussVisibleStornMixin(object):

    def make_gen(self, latent_sample):
        return neural_dists.FastDropoutRnnDiagGauss(
            latent_sample,
            n_inpt=self.n_latent + self.n_inpt,
            n_hiddens=self.n_hiddens_gen,
            n_output=self.n_inpt,
            hidden_transfers=self.gen_transfers,
            p_dropout_inpt=self.p_dropout_inpt,
            p_dropout_hiddens=self.p_dropout_hiddens,
            p_dropout_hidden_to_out=self.p_dropout_hidden_to_out,
            declare=self.parameters.declare)


class ConstVarGaussVisibleStornMixin(object):

    shared_std = False
    fixed_std = None

    def make_gen(self, latent_sample):
        return neural_dists.FastDropoutRnnConstDiagGauss(
            latent_sample,
            n_inpt=self.n_latent + self.n_inpt,
            n_hiddens=self.n_hiddens_gen,
            n_output=self.n_inpt,
            hidden_transfers=self.gen_transfers,
            p_dropout_inpt=self.p_dropout_inpt,
            p_dropout_hiddens=self.p_dropout_hiddens,
            p_dropout_hidden_to_out=self.p_dropout_hidden_to_out,
            shared_std=self.shared_std,
            fixed_std=self.fixed_std,
            declare=self.parameters.declare)


class GaussLatentStornMixin(object):

    distribution_klass = neural_dists.FastDropoutRnnDiagGauss

    def make_prior(self, sample):
        return NormalGauss(sample.shape)

    def make_recog(self, inpt, recog=None):
        return self.distribution_klass(
            inpt,
            n_inpt=self.n_inpt,
            n_hiddens=self.n_hiddens_recog,
            n_output=self.n_latent,
            hidden_transfers=self.recog_transfers,
            p_dropout_inpt='parameterized',
            p_dropout_hiddens=['parameterized' for _ in self.n_hiddens_recog],
            p_dropout_hidden_to_out='parameterized',
            declare=self.parameters.declare)


class GaussLatentBiStornMixin(GaussLatentStornMixin):

    distribution_klass = neural_dists.FastDropoutBiRnnDiagGauss


from breze.arch.construct.neural.base import wild_reshape
from breze.arch.construct.layer.varprop import simple as vp_simple


class TimeDependentGaussLatent(GaussLatentStornMixin):

    p_dropout = 0.25

    def _make_gaus_inputs(self, sample, recog, n_output=None, transfer='identity'):

        n_inpt = recog.rnn.layers[-4].n_output

        if n_output is None:
            n_output = recog.n_output

        n_time_steps, _, _ = recog.inpt.shape

        x_mean, x_var = recog.rnn.layers[-3].outputs   # mean and variance of the hidden state
        x_mean_flat = wild_reshape(x_mean, (-1, n_inpt))
        x_var_flat = wild_reshape(x_var, (-1, n_inpt))
        fd = vp_simple.FastDropout(
            x_mean_flat, x_var_flat, self.p_dropout)

        x_mean_flat, x_var_flat = fd.outputs

        affine = vp_simple.AffineNonlinear(
            x_mean_flat, x_var_flat, n_inpt, n_output, transfer,
            declare=self.parameters.declare)
        output_mean_flat, output_var_flat = affine.outputs

        output_mean = wild_reshape(output_mean_flat, (n_time_steps, -1, n_output))
        output_var = wild_reshape(output_var_flat, (n_time_steps, -1, n_output))
        return output_mean, output_var

    def make_prior(self, sample, recog):
        output_mean, output_var = self._make_gaus_inputs(sample, recog)
        return DiagGauss(output_mean, output_var)


class TimeDependentBasisDiagGauss(TimeDependentGaussLatent):

    num_basis = 10

    def make_prior(self, sample, recog):

        n_time_steps, _, _ = recog.inpt.shape
        out_dims = recog.n_output
        n_output = self.num_basis * out_dims

        mean, var = self._make_gaus_inputs(sample, recog, n_output)

        mean = wild_reshape(mean, (n_time_steps, -1, self.num_basis, out_dims))
        var = wild_reshape(var, (n_time_steps, -1, self.num_basis, out_dims))

        basis_indices = T.constant(np.linspace(0, 1, self.num_basis), dtype=theano.config.floatX)
        t = T.fscalar('t')
        timesteps = T.arange(0, 1, 1. / n_time_steps)

        def times_basis(tens, tt, b):
            basis = T.exp(-(tt-b)**2 / 2)
            return T.dot(basis / basis.sum(), tens)

        mean, _ = theano.scan(times_basis, sequences=[mean, timesteps], non_sequences=basis_indices)
        var, _ = theano.scan(times_basis, sequences=[var, timesteps], non_sequences=basis_indices)

        # return FullGauss(mean, var)
        return DiagGauss(mean, var)


class TimeDependentRankOneGauss(TimeDependentGaussLatent):

    diag_constant = 1

    def make_prior(self, sample, recog):
        mean, var = self._make_gaus_inputs(sample, recog)

        if self.diag_constant:
            diag = self.diag_constant
        else:
            diag, _ = self._make_gaus_inputs(sample, recog, transfer='sigmoid')

        return RankOneGauss(mean, diag, var)


class StochasticRnn(GenericVariationalAutoEncoder):

    sample_dim = 1,
    theano_optimizer = optdb.query(theano.gof.Query(
        include=['fast_run'], exclude=['scan_eqopt1', 'scan_eqopt2']))
    mode = theano.Mode(linker='cvm', optimizer=theano_optimizer)

    def __init__(self, n_inpt, n_hiddens_recog, n_latent, n_hiddens_gen,
                 recog_transfers, gen_transfers,
                 p_dropout_inpt=.1, p_dropout_hiddens=.1,
                 p_dropout_hidden_to_out=None,
                 p_dropout_shortcut=None,
                 use_imp_weight=False,
                 batch_size=None, optimizer='rprop',
                 max_iter=1000, verbose=False):

        self.n_hiddens_recog = n_hiddens_recog
        self.n_hiddens_gen = n_hiddens_gen

        self.recog_transfers = recog_transfers
        self.gen_transfers = gen_transfers

        self.p_dropout_inpt = p_dropout_inpt
        self.p_dropout_hiddens = p_dropout_hiddens
        self.p_dropout_hidden_to_out = p_dropout_hidden_to_out
        self.p_dropout_shortcut = p_dropout_shortcut

        super(StochasticRnn, self).__init__(
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

    def initialize(self,
                   par_std=1, par_std_affine=None, par_std_rec=None,
                   par_std_in=None,
                   sparsify_affine=None, sparsify_rec=None,
                   spectral_radius=None):
        climin.initialize.randomize_normal(self.parameters.data, 0, par_std)
        all_layers = self.vae.recog.rnn.layers + self.vae.gen.rnn.layers
        P = self.parameters
        for i, layer in enumerate(all_layers):
            if isinstance(layer, FDRecurrent):
                p = P[layer.weights]
                if par_std_rec:
                    climin.initialize.randomize_normal(p, 0, par_std_rec)
                if sparsify_rec:
                    climin.initialize.sparsify_columns(p, sparsify_rec)
                if spectral_radius:
                    climin.initialize.bound_spectral_radius(p, spectral_radius)
                P[layer.initial_mean][...] = 0
                P[layer.initial_std][...] = 1
            if isinstance(layer, AffineNonlinear):
                p = self.parameters[layer.weights]
                if par_std_affine:
                    if i == 0 and par_std_in:
                        climin.initialize.randomize_normal(p, 0, par_std_in)
                    else:
                        climin.initialize.randomize_normal(p, 0, par_std_affine)
                if sparsify_affine:
                    climin.initialize.sparsify_columns(p, sparsify_affine)

                self.parameters[layer.bias][...] = 0

    #def _make_gen_hidden(self):
    #    hidden_exprs = [T.concatenate(i.recurrent.outputs, 2)
    #                    for i in self.vae.gen.hidden_layers]

    #    return self.function(['inpt'], hidden_exprs)

    #def gen_hidden(self, X):
    #    if getattr(self, 'f_gen_hiddens', None) is None:
    #        self.f_gen_hiddens = self._make_gen_hidden()
    #    return self.f_gen_hiddens(X)

    def sample(self, n_time_steps, prefix=None, visible_map=False):
        if prefix is None:
            raise ValueError('need to give prefix')

        if not hasattr(self, 'f_gen'):
            vis_sample = self.vae.gen.inpt

            inpt_m1 = T.tensor3('inpt_m1')
            inpt_m1.tag.test_value = np.zeros((3, 2, self.n_inpt))

            latent_prior_sample = T.tensor3('latent_prior_sample')
            latent_prior_sample.tag.test_value = np.zeros((3, 2, self.n_latent))

            gen_inpt_sub = T.concatenate([latent_prior_sample, inpt_m1], axis=2)

            vis_sample = theano.clone(
                vis_sample,
                {self.vae.gen.inpt: gen_inpt_sub}
            )

            gen_out_sub = theano.clone(
                self.vae.gen.rnn.output, {self.vae.gen.inpt: gen_inpt_sub})
            self._f_gen_output = self.function(
                [inpt_m1, latent_prior_sample],
                gen_out_sub, mode='FAST_COMPILE',
                on_unused_input ='warn')

            out = self.vae.gen.sample() if not visible_map else self.vae.gen.maximum
            self._f_visible_sample_by_gen_output = self.function(
                [self.vae.gen.rnn.output], out,
                on_unused_input ='warn')

            def f_gen(inpt_m1, latent_prior_sample):
                rnn_out = self._f_gen_output(inpt_m1, latent_prior_sample)
                return self._f_visible_sample_by_gen_output(rnn_out)

            self.f_gen = f_gen

        prefix_length = prefix.shape[0]
        S = np.empty(
            (prefix.shape[0] + n_time_steps, prefix.shape[1],
             prefix.shape[2])
        ).astype(theano.config.floatX)
        S[:prefix_length][...] = prefix
        latent_samples = np.zeros(
            (prefix.shape[0] + n_time_steps, prefix.shape[1], self.n_latent)
            ).astype(theano.config.floatX)
        latent_samples[prefix_length:] = np.random.standard_normal(
            (n_time_steps, prefix.shape[1], self.n_latent))
        for i in range(n_time_steps):
            p = self.f_gen(
                S[:prefix_length + i], latent_samples[:prefix_length + i]
            )[-1, :, :self.n_inpt]
            S[prefix_length + i] = p

        return S[prefix_length:]
