# -*- coding: utf-8 -*-

import theano
import theano.tensor as T
import numpy as np

from breze.arch.util import wild_reshape


def assert_no_time(X):
    if X.ndim == 2:
        return X
    if X.ndim != 3:
        raise ValueError('ndim must be 2 or 3, but it is %i' % X.ndim)
    return wild_reshape(X, (-1, X.shape[2]))


def recover_time(X, time_steps):
    return wild_reshape(X, (time_steps, -1, X.shape[1]))


def normal_logpdf(xs, means, vrs):
    energy = -(xs - means) ** 2 / (2 * vrs)
    partition_func = -T.log(T.sqrt(2 * np.pi * vrs))
    return partition_func + energy


class Distribution(object):

    def __init__(self, rng=None):
        if rng is None:
            self.rng = T.shared_randomstreams.RandomStreams()
        else:
            self.rng = rng

    def sample(self, epsilon=None):
        raise NotImplemented()

    def nll(self, X, inpt=None):
        raise NotImplemented()

    def entropy(self):
        raise NotImplemented()


class NormalizingFlow(Distribution):
    def __init__(self, initial_dist, neglogdet, rng=None):
        self.initial_dist = initial_dist
        self.neglogdet = neglogdet
        super(NormalizingFlow, self).__init__(rng)


def extract_flow_pars(parameters, flow_transfers, n_state):
    flow_pars = []
    start = 0
    for flow_transfer in flow_transfers:
        if flow_transfer == 'planar':
            w = parameters[:,               start
                             :n_state * 1 + start]
            u = parameters[:, n_state * 1 + start
                             :n_state * 2 + start]
            b = parameters[:, n_state * 2 + start
                             :n_state * 2 + start + 1]
            flow_pars.append((w, u, b))
            start += n_state * 2 + 1
        elif flow_transfer == 'radial':
            ref = parameters[:,             start
                                 :n_state + start]
            alpha = parameters[:, n_state + start
                                 :n_state + start + 1]
            beta = parameters[:,  n_state + start + 1
                                 :n_state + start + 2]
            flow_pars.append((ref, alpha, beta))
            start += n_state + 2
        else:
            # TODO: set a default case, probably planar
            raise NotImplementedError
    return flow_pars


def softplus(x):
    return T.log(1.0 + T.exp(x))


def reparametrize(flow_par, flow_transfer):
    m = lambda x: - 1.0 + T.log(1.0 + T.exp(x))
    if flow_transfer == 'planar':
        w, u, b = flow_par
        u = u + ((-1 + softplus((w * u).sum(axis=1))
                    - (w * u).sum(axis=1)).dimshuffle(0, 'x') * w
                    / (w**2).sum(1).dimshuffle(0, 'x'))
        return w, u, b
    elif flow_transfer == 'radial':
        ref, alpha, beta = flow_par
        alpha = T.exp(alpha)
        beta = -alpha + softplus(beta)
        return ref, alpha, beta
    else:
        # TODO: set a default case, probably planar
        raise NotImplementedError


def flow(z, flow_par, flow_transfer, n_state=None):
    f = lambda x: T.tanh(x)
    df = lambda x: 1.0 - T.tanh(x) ** 2
    h = lambda x, y: 1.0 / (x + y)
    dh = lambda x, y: -1.0 / (x + y)**2
    if flow_transfer == 'planar':
        w, u, b = flow_par
        delta_nld = - T.log( abs(1.0 + (u * df( (z * w).sum(1)
                             + b.ravel()).dimshuffle(0, 'x') * w).sum(1) ))
        delta_z = u * f((z * w).sum(1) + b.ravel()).dimshuffle(0, 'x')
    elif flow_transfer == 'radial':
        ref, alpha, beta = flow_par
        r = T.sqrt(((z - ref)**2).sum(1)).dimshuffle(0,'x')
        delta_nld = - T.log(abs(
                (1.0 + beta * h(alpha, r)) ** (n_state - 1)
                * (1.0 + beta * h(alpha, r) + beta * dh(alpha, r) * r)
            )).ravel()

        delta_z = T.Rebroadcast((1,True))(beta * h(alpha,r)) * (z - ref)
    else:
        # TODO: set a default case, probably planar
        raise NotImplementedError
    return delta_z, delta_nld


class GenericNormalizingFlow(NormalizingFlow):
    def __init__(self, flow_transfers, n_state, initial_dist, parameters,
                 rng=None):
        self.flow_transfers = flow_transfers
        self.n_state = n_state

        self.z_0 = initial_dist.sample()
        if self.z_0.ndim == 3:
            self.z = assert_no_time(self.z_0)
            parameters = assert_no_time(parameters)
        else:
            self.z = self.z_0

        neglogdet = 0.0

        flow_pars = extract_flow_pars(parameters, self.flow_transfers,
                                      self.n_state)

        for flow_par, flow_transfer in zip(flow_pars, flow_transfers):
            flow_par = reparametrize(flow_par, flow_transfer)
            delta_z, delta_nld = flow(self.z, flow_par, flow_transfer,
                                      self.n_state)
            self.z += delta_z
            neglogdet += delta_nld

        if self.z_0.ndim == 3:
            self.z = recover_time(self.z, self.z_0.shape[0])

        super(GenericNormalizingFlow, self).__init__(initial_dist, neglogdet,
                                                    rng)

    def sample(self, epsilon=None):
        # TODO: handle new sampling without using only samples from init
        return self.z


class PlanarNormalizingFlow(NormalizingFlow):
    """
    joint_with - if not None, computes a joint planar flow over (initial_dist.sample(), joint_with)
                by reusing the same parameters. The jacobian of such a flow has a block structure,
                and the det-jacobian w.r.t initial_dist and joint_with is the same if number of
                dimensions of each of them is the same - it saves the partial neglogdet in
                self.partial_neglogdet
    """
    def __init__(self, n_layer, n_state, initial_dist=None, parameters=None, rng=None, joint_with=None, sample=None):
        self.n_layer = n_layer
        self.n_state = n_state

        f = lambda x: T.tanh(x)
        df = lambda x: 1.0 - T.tanh(x) ** 2
        m = lambda x: - 1.0 + T.log(1.0 + T.exp(x))
        self.initial_dist = initial_dist

        if sample is None:
            self.z_0 = self.initial_dist.sample()
        else:
            self.z_0 = sample
        if self.z_0.ndim == 3:
            self.z = assert_no_time(self.z_0)
            parameters = assert_no_time(parameters)
        else:
            self.z = self.z_0

        self.joint_flow = joint_with is not None
        if self.joint_flow:
            self.y_0 = joint_with
            if joint_with.ndim == 3:
                self.y = assert_no_time(self.y_0)
            else:
                self.y = self.y_0

            self.partial_dims = self.z.shape[-1]
            self.z = T.concatenate((self.z, self.y), axis=1)
            partial_neglogdet = 0.0

        neglogdet = 0.0
        for i in range(self.n_layer):
            w = parameters[:,               (n_state * 2 + 1) * i
                             :n_state * 1 + (n_state * 2 + 1) * i]
            u = parameters[:, n_state * 1 + (n_state * 2 + 1) * i
                             :n_state * 2 + (n_state * 2 + 1) * i]
            b = parameters[:, n_state * 2 + (n_state * 2 + 1) * i
                             :n_state * 2 + (n_state * 2 + 1) * i + 1]

            # orthogonalization of w and u
            u = u + ((m((w * u).sum(axis=1))
                    - (w * u).sum(axis=1)).dimshuffle(0, 'x') * w
                    / (w**2).sum(1).dimshuffle(0, 'x'))

            # joint flow uses the same parameters for both parts of the flow
            # no need to tile b since it's a scalar
            if self.joint_flow:
                w, u = (T.tile(i, (1, 2)) for i in(w, u))

            det_part = u * df( (self.z * w).sum(1)
                           + b.ravel()).dimshuffle(0, 'x') * w

            neglogdet -= T.log(abs(1.0 + det_part.sum(1)))

            if self.joint_flow:
                det_part = det_part[:, :self.partial_dims]
                partial_neglogdet -= T.log(abs(1.0 + det_part.sum(1)))

            self.z = (self.z
                + u * f((self.z * w).sum(1) + b.ravel()).dimshuffle(0, 'x'))

        if self.joint_flow:
            self.partial_neglogdet = partial_neglogdet
            self.y = self.z[:, self.partial_dims:]
            self.z = self.z[:, :self.partial_dims]
            if self.y_0.ndim == 3:
                self.y = recover_time(self.y, self.y_0.shape[0])

        if self.z_0.ndim == 3:
            self.z = recover_time(self.z, self.z_0.shape[0])

        super(PlanarNormalizingFlow, self).__init__(initial_dist, neglogdet,
                                                    rng)

    def sample(self, epsilon=None):
        # TODO: handle new sampling without using only samples from init
        if epsilon is not None:
            z = theano.clone(self.z, {self.z_0: self.initial_dist.sample(epsilon=epsilon)})
        else:
            z = self.z

        return z


class RadialNormalizingFlow(NormalizingFlow):
    def __init__(self, n_layer, n_state, initial_dist, parameters, rng=None):
        self.n_layer = n_layer
        self.n_state = n_state

        h = lambda x, y: 1.0 / (x + y)
        dh = lambda x, y: -1.0 / (x + y)**2
        m = lambda x: T.log(1.0 + T.exp(x))
        self.z_0 = initial_dist.sample()
        if self.z_0.ndim == 3:
            self.z = assert_no_time(self.z_0)
            parameters = assert_no_time(parameters)
        else:
            self.z = self.z_0
        neglogdet = 0.0

        for i in range(self.n_layer):
            # reference point z_0 in the paper, renamed to ref to avoid
            # confusion with initial sample z_0 here.

            ref = parameters[:,             (n_state + 2) * i
                                 :n_state + (n_state + 2) * i]
            alpha = parameters[:, n_state + (n_state + 2) * i
                                 :n_state + (n_state + 2) * i + 1]
            beta = parameters[:,  n_state + (n_state + 2) * i + 1
                                 :n_state + (n_state + 2) * i + 2]

            alpha = T.exp(alpha)
            # alpha = T.log(1.0 + T.exp(alpha))
            beta = -alpha + m(beta)
            # beta = -alpha + m(beta + alpha)

            r = T.sqrt(((self.z - ref)**2).sum(1)).dimshuffle(0,'x')

            neglogdet -= T.log(abs(
                (1.0 + beta * h(alpha, r)) ** (self.n_state - 1)
                * (1.0 + beta * h(alpha, r) + beta * dh(alpha, r) * r)
            ))

            deltaz = T.Rebroadcast((1,True))(beta * h(alpha,r)) * (self.z - ref)
            self.z = (self.z + deltaz)

        if self.z_0.ndim == 3:
            self.z = recover_time(self.z, self.z_0.shape[0])

        super(RadialNormalizingFlow, self).__init__(initial_dist,
                    T.Rebroadcast((1, True))(neglogdet), rng)

    def sample(self, epsilon=None):
        # TODO: handle new sampling without using only samples from init
        return self.z


class DiagGauss(Distribution):

    def __init__(self, mean, var, rng=None):
        self.mean = mean

        # This allows to use var with shape (1, 1, n)
        self.var = T.fill(mean, var)

        self.stt = T.concatenate((mean, self.var), -1)
        self.maximum = self.mean
        super(DiagGauss, self).__init__(rng)

    def sample(self, epsilon=None):
        mean_flat = assert_no_time(self.mean)
        var_flat = assert_no_time(self.var)

        if epsilon is None:
            noise = self.rng.normal(size=mean_flat.shape)
        else:
            noise = epsilon

        sample = mean_flat + T.sqrt(var_flat) * noise
        if self.mean.ndim == 3:
            return recover_time(sample, self.mean.shape[0])
        else:
            return sample

    def nll(self, X, inpt=None):
        var_offset = 1e-4
        var = self.var
        var += var_offset
        residuals = X - self.mean
        weighted_squares = -(residuals ** 2) / (2 * var)
        normalization = T.log(T.sqrt(2 * np.pi * var))
        ll = weighted_squares - normalization
        return -ll

    def entropy(self):
        return 0.5 * T.log(2.0 * np.pi * np.e * self.var)


class NormalGauss(Distribution):

    def __init__(self, shape, rng=None):
        self.shape = shape
        self.mean = T.zeros(shape)
        self.var = T.ones(shape)
        self.stt = T.concatenate((self.mean, self.var), -1)
        self.maximum = self.mean
        super(NormalGauss, self).__init__(rng)

    def sample(self):
        return self.rng.normal(size=self.shape)

    def nll(self, X, inpt=None):
        X_flat = X.flatten()
        nll = -normal_logpdf(X_flat, self.mean.flatten(), self.var.flatten())
        return nll.reshape(X.shape)

    def entropy(self):
        return 0.5 * T.log(2.0 * np.pi * np.e * self.var)


class Bernoulli(Distribution):

    def __init__(self, rate, rng=None):
        self.rate = rate
        self.stt = rate
        self.maximum = self.rate > 0.5
        super(Bernoulli, self).__init__(rng)

    def sample(self, epsilon=None):
        if epsilon is None:
            noise = self.rng.uniform(size=self.rate.shape)
        else:
            noise = epsilon
        sample = noise < self.rate
        return sample

    def nll(self, X, inpt=None):
        rate = self.rate
        rate *= 0.999
        rate += 0.0005
        return -(X * T.log(rate) + (1 - X) * T.log(1 - rate))


class Categorical(Distribution):
    """Class representing a Categorical distribution.

    Attributes
    ----------

    probs: Theano variable
        Has the same shape as the distribution and contains the probability of
        the element being 1 and all others being 0. I.e. rows sum up to 1.

    stt : Theano variable.
        Same as ``probs``.

    maximum : Theano variable.
        Maximum of the distribution.

    rng : Theano RandomStreams object.
        Random number generator to draw samples from the distribution from.
    """

    def __init__(self, probs, rng=None):
        """Initialize a Categorical object.

        Parameters
        ----------

        probs : Theano variable
            Gives the shape of the distribution and contains the probability of
            an element being 1, where only one of a row will be 1. Rows sum up
            to 1.

        rng : Theano RandomStreams object, optional.
            Random number generator to draw samples from the distribution from.
        """
        self.probs = probs
        self.stt = probs
        self.maximum = T.eye(probs.shape[1])[T.argmax(probs, 1)]

        super(Categorical, self).__init__(rng)

    def sample(self):
        """Return a sample of the distribution.

        Returns
        -------

        S : Theano variable
            Has the same shape as the distribution, only one and exactly one
            element per row will be set to 1."""
        return self.rng.multinomial(pvals=self.probs)

    def nll(self, X):
        """Return the negative log-likelihood of an observation ``X`` under the
        distribution.

        Parameters
        ----------

        X : Theano variable
            Has to have the same shape as the distribution.

        Returns
        -------

        L : Theano variable.
            Has the same shape as ``X``, i.e. coordinate wise result.
        """
        return loss.cat_ce(X, self.probs)


class ApproxSpikeAndSlab(Distribution):
    """Class representing an approximate spike and slab distribution.

    The distribution is approximate with a Gaussian scale mixture, consisting of
    two components. A scale mixture is a mixture of Gaussians where each of the
    components has a zero mean.

    Attributes
    ---

    spike_ratio : Theano variable.
        Value between 0 and 1 which gives the prior probability of a spike.

    spike_std : Theano variable.
        Standard deviation of the spike. Can be negative, positiveness will be
        ensured.

    slab_std : Theano variable.
        Standard deviationof the slab. Can be negative, positiveness will be
        ensured.

    rng : Theano RandomStates object.
        Random number generator for the expressions

    """
    def __init__(self, spike_ratio, spike_std, slab_std, rng=None):
        self.spike_ratio = spike_ratio
        self.spike_std = spike_std
        self.slab_std = slab_std

    def sample(self):
        # TODO implement
        raise NotImplemented()

    def nll(self, X, inpt=None):
        var_offset = 1e-4
        var = self.spike_std ** 2
        var += var_offset
        weighted_squares = -(X ** 2) / (2 * var)
        normalization = T.log(T.sqrt(2 * np.pi * var))
        spike_likelihood = T.exp(weighted_squares - normalization)

        var = self.slab_std ** 2
        var += var_offset
        weighted_squares = -(X ** 2) / (2 * var)
        normalization = T.log(T.sqrt(2 * np.pi * var))
        slab_likelihood = T.exp(weighted_squares - normalization)

        return T.log(self.spike_ratio * spike_likelihood
                     + (1 - self.spike_ratio) * slab_likelihood + 1e-8)
