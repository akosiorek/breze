import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
theano_flags = os.environ['THEANO_FLAGS'] if 'THEANO_FLAGS' in os.environ else ''
os.environ['THEANO_FLAGS'] = 'device=cpu,' + theano_flags
os.environ['GNUMPY_IMPLICIT_CONVERSION'] = 'allow'

from unittest import TestCase
import theano
import theano.tensor as T
from breze.arch.construct.layer.distributions import RankOneGauss, DiagGauss
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from numpy import random, log, trace
from numpy.linalg import inv, det
from scipy.stats import multivariate_normal, entropy

from breze.arch.construct.layer.distributions import DiagGauss, RankOneGauss
from breze.arch.component.misc import inter_gauss_kl


def kl_div(p, q):
    return inter_gauss_kl(p.mean, p.var, q.mean, q.var, u2=q.u, eta2=q.eta)


def kl_gaussian(m1, cov1, m2, cov2):
    m12 = m1 - m2
    icov2 = inv(cov2)
    kl = 0
    kl += log(det(cov2)/det(cov1))
    kl += - cov2.shape[-1] + trace(cov1.dot(icov2))
    kl += m12.dot(icov2).dot(m12)
    return kl / 2


def kl_gaussian_tensor(m1, cov1, m2, cov2, u=None):
    kl = 0
    shape = m1.shape
    for i in xrange(shape[0]):
        for j in xrange(shape[1]):
            m11, cov11 = m1[i, j, :], np.diag(cov1[i, j, :])
            m22, cov22 = m2[i, j, :], np.diag(cov2[i, j, :])
            if u is not None:
                uu = u[i, j, :][:, np.newaxis]
                cov22 += uu.dot(uu.T)
            kl += kl_gaussian(m11, cov11, m22, cov22)
    return kl


class TestDiagRankOneKL(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mean_T, cls.var_T, cls.u_T = (T.tensor3(i) for i in ('mean', 'var', 'u'))
        cls.mean_diag_T, cls.var_diag_T = (T.tensor3(i) for i in ('mean2', 'var2'))
        cls.rank_one_gaus = RankOneGauss(cls.mean_T, cls.var_T, cls.u_T, eps=0)
        cls.diag_gaus = DiagGauss(cls.mean_diag_T, cls.var_diag_T)

        cls.n = int(1e4)
        cls.dims = 2
        cls.shape = [cls.dims] * 3
        cls.floatx = theano.config.floatX

    def setUp(self):
        floatx = self.floatx
        self.m1, self.m2, self.u = (random.randn(*self.shape).astype(floatx) for _ in xrange(3))
        self.v1, self.v2 = (random.randn(*self.shape).astype(floatx) ** 2 for _ in xrange(2))

    def _approximate_gauss_kl(self):
        args = self.mean_diag_T, self.var_diag_T
        sample_diag = self.diag_gaus.sample()
        sample_foo = theano.function(args, sample_diag)

        mvn_diag, mvn_rank = (np.empty(self.shape[:-1], dtype=np.object) for _ in xrange(2))
        for i in xrange(self.shape[0]):
            for j in xrange(self.shape[1]):
                m = self.m1[i, j, :]
                v = self.v1[i, j, :]
                u = self.u[i, j, :, np.newaxis]

                cov_diag = np.diagflat(v)
                cov_rank = u.dot(u.T) + cov_diag

                mvn_diag[i, j] = multivariate_normal(m, cov_diag).pdf
                mvn_rank[i, j] = multivariate_normal(m, cov_rank).pdf

        samples = np.empty(list(self.shape) + [self.n])
        for n in xrange(self.n):
            s_diag = sample_foo(self.m1, self.v1)
            samples[..., n] = s_diag

        approximate_kl = 0
        for i in xrange(self.shape[0]):
            for j in xrange(self.shape[1]):
                pdfs = mvn_diag[i, j](samples[i, j].T), mvn_rank[i, j](samples[i, j].T)
                approximate_kl += log(pdfs[0] / pdfs[1]).mean()
        return approximate_kl

    def test_rank_zero_u_vs_diag(self):
        args = [self.mean_diag_T, self.var_diag_T, self.mean_T, self.var_T, self.u_T]
        diag_rank = theano.function(args,
                                    kl_div(self.diag_gaus, self.rank_one_gaus), on_unused_input='warn')

        u = np.zeros_like(self.u)
        result = diag_rank(self.m1, self.v1, self.m1, self.v1, u)

        self.assertEqual(result.sum(), 0)
        self.assertFalse(np.isnan(result).any())
        assert_array_equal(result.shape, u.shape)

    def test_rank_nonzero_u_vs_diag(self):
        args = [self.mean_diag_T, self.var_diag_T, self.mean_T, self.var_T, self.u_T]
        diag_rank = theano.function(args,
                                    kl_div(self.diag_gaus, self.rank_one_gaus), on_unused_input='warn')

        result = diag_rank(self.m1, self.v1, self.m1, self.v1, self.u)

        self.assertGreater(result.sum(), 0)
        self.assertFalse(np.isnan(result).any())

    def test_against_bruteforce(self):
        args = [self.mean_diag_T, self.var_diag_T, self.mean_T, self.var_T, self.u_T]
        diag_rank = theano.function(args,
                                    kl_div(self.diag_gaus, self.rank_one_gaus), on_unused_input='warn')

        computed_kl = diag_rank(self.m1, self.v1, self.m2, self.v2, self.u).sum()
        brute_kl = kl_gaussian_tensor(self.m1, self.v1, self.m2, self.v2, self.u)
        # approximate_kl = self._approximate_gauss_kl()
        # print approximate_kl
        # assert_allclose(brute_kl, approximate_kl, 0.02)

        # print 'computed kl:', computed_kl
        # print 'true kl:', brute_kl
        # print 'shape:', computed_kl.shape
        assert_allclose(computed_kl, brute_kl, 0.02, err_msg='Computed KL = {} is incorrect'.format(computed_kl))

