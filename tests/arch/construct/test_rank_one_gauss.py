import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
theano_flags = os.environ['THEANO_FLAGS'] if 'THEANO_FLAGS' in os.environ else ''
os.environ['THEANO_FLAGS'] = 'device=cpu,' + theano_flags
os.environ['GNUMPY_IMPLICIT_CONVERSION'] = 'allow'


import theano
import theano.tensor as T
from breze.arch.construct.layer.distributions import RankOneGauss
import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import multivariate_normal

# np.set_printoptions(precision=4, linewidth=120)


class TestRankOneGauss():

    @classmethod
    def setup_class(cls):
        m, v, u = (T.tensor3(i) for i in ('mean', 'var', 'u'))
        cls.mean_T = m
        cls.var_T = v
        cls.u_T = u
        cls.gaus = RankOneGauss(m, v, u, eps=0)
        cls.n = int(1e4)
        cls.dims = 2
        cls.shape = [cls.dims] * 3
        # cls.shape = [1, 1, 2]
        cls.floatx = theano.config.floatX

    def setup(self):
        floatx = self.floatx
        self.mean = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=floatx).reshape(*self.shape)
        self.var = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=floatx).reshape(*self.shape)
        self.u = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=floatx).reshape(*self.shape)

        # self.mean = np.array([1, 2], dtype=floatx).reshape(*self.shape)
        # self.var = np.array([1, 2], dtype=floatx).reshape(*self.shape)
        # self.u = np.array([1, 2], dtype=floatx).reshape(*self.shape)

    def cov(self):
        u = self.u.reshape(-1, 1)
        return np.diagflat(self.var) + u.dot(u.T)

    def test_sample(self):
        foo = theano.function([self.mean_T, self.var_T, self.u_T], self.gaus.sample())
        X = np.zeros((self.dims ** 3, self.n), dtype=self.floatx)
        for i in xrange(self.n):
            X[:, i] = foo(self.mean, self.var, self.u).flatten()

        mean = X.mean(axis=1).reshape(self.mean.shape)
        cov = np.cov(X)
        expected_cov = self.cov()

        assert_allclose(mean, self.mean, 0.05)
        for i in xrange(4):
            c = cov[i*2:(i+1)*2, i*2:(i+1)*2]
            ec = expected_cov[i*2:(i+1)*2, i*2:(i+1)*2]

            print c
            print ec
            assert_allclose(c, ec, 0.05)

    def test_nll(self):

        x_T = T.tensor3('inpt')
        nll_expr = self.gaus.nll(x_T)
        foo = theano.function([self.mean_T, self.var_T, self.u_T, x_T], nll_expr, on_unused_input='warn')
        x = np.ones(self.shape, dtype=self.floatx)

        cov = self.cov()
        residual = (x - self.mean)
        nll = foo(self.mean, self.var, self.u, x)
        nll = nll.reshape(2, 2, -1)

        for i in xrange(self.dims**2):
            d = self.dims
            c = cov[i*d:(i+1)*d, i*d:(i+1)*d]
            rc = nll[i // 2, i % 2].sum()
            r = residual[i // 2, i % 2].reshape(d, 1)

            exponent = r.T.dot(np.linalg.inv(c)).dot(r)
            norm = (2 * np.pi) ** d * np.linalg.det(c)
            expected_nll = 0.5 * (exponent + np.log(norm)).flatten()

            assert_allclose(rc, expected_nll)
