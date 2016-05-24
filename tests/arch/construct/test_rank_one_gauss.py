import theano
import theano.tensor as T
from breze.arch.construct.layer.distributions import RankOneGauss
import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import multivariate_normal

# np.set_printoptions(precision=4, linewidth=120)


class TestRankOneGauss(object):

    @classmethod
    def setup_class(cls):
        cls.mean_T, cls.var_T, cls.u_T = (T.tensor3(i) for i in ('mean', 'var', 'u'))
        cls.gaus = RankOneGauss(cls.mean_T, cls.var_T, cls.u_T, eps=0)
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

    def test_time_cov_factor(self):
        inpts = [T.vector(i) for i in 'var u x'.split()]
        foo = theano.function(inpts, self.gaus._times_cov_factor(*inpts))

        for i in xrange(self.shape[0]):
            for j in xrange(self.shape[1]):

                u = self.u[i, j, :]
                v = self.var[i, j, :]
                v = np.ones(v.shape, dtype=self.floatx)
                u = np.ones(u.shape, dtype=self.floatx)
                x = np.random.randn(self.dims).astype(self.floatx)

                result = (foo(v, u, x) ** 2).sum()
                x = x.reshape(1, self.dims)
                u = u.reshape(1, self.dims)
                cov = u.T.dot(u) + np.diagflat(v)
                expected = x.dot(cov).dot(x.T)
                print cov
                print result, expected
                assert_allclose(result, expected, 1e-6)

    def test_cov_factor(self):
        inpts = [T.vector(i) for i in 'var u'.split()]
        foo = theano.function(inpts, self.gaus._cov_factor(*inpts))

        for i in xrange(self.shape[0]):
            for j in xrange(self.shape[1]):

                u = self.u[i, j, :]
                v = self.var[i, j, :]

                result = foo(v, u).reshape(2, 2)
                result = result.T.dot(result)
                u = u.reshape(self.dims, 1)
                cov = u.dot(u.T) + np.diagflat(v)
                assert_allclose(result, cov, 1e-6)

    def test_sample(self):
        foo = theano.function([self.mean_T, self.var_T, self.u_T], self.gaus.sample(), on_unused_input='warn')
        X = np.zeros((self.dims ** 3, self.n), dtype=self.floatx)
        for i in xrange(self.n):
            X[:, i] = foo(self.mean, self.var, self.u).flatten()

        mean = X.mean(axis=1).reshape(self.mean.shape)
        expected_cov = self.cov()

        assert_allclose(mean, self.mean, 0.05)
        print 'mean =', mean
        X -= mean.reshape(-1, 1)
        for i in xrange(4):
            c = np.cov(X[i*2:(i+1)*2, :])
            ec = expected_cov[i*2:(i+1)*2, i*2:(i+1)*2]

            print c
            print ec
            assert_allclose(c, ec, 0.2)

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
