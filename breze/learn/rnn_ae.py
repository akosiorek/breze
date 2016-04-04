import numpy as np
import theano
from theano import tensor as T

from breze.learn.rnn import BaseRnn
from breze.arch.util import ParameterSet, Model
from breze.arch.model.rnn import rnn
from breze.learn.base import UnsupervisedBrezeWrapperBase
from breze.arch.component.common import supervised_loss


class RnnAE(Model, UnsupervisedBrezeWrapperBase):

    def __init__(self, n_inpt, n_hiddens, n_code, hidden_transfers,
                 code_transfer='identity',
                 out_transfer='identity',
                 loss='squared',
                 batch_size=None,
                 optimizer='rprop',
                 imp_weight=False,
                 max_iter=1000,
                 verbose=False):

        self.n_inpt = n_inpt
        self.n_hiddens = n_hiddens
        self.n_code = n_code
        self.hidden_transfers = hidden_transfers
        self.code_transfer = code_transfer
        self.out_transfer = out_transfer
        self.loss = loss
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.imp_weight = imp_weight
        self.max_iter = max_iter
        self.verbose = verbose

        super(RnnAE, self).__init__()

    def _init_pars(self):
        spec_encode = rnn.parameters(self.n_inpt, self.n_hiddens, self.n_code, prefix='encode_')
        spec_decode = rnn.parameters(self.n_code, list(reversed(self.n_hiddens)), self.n_inpt, prefix='decode_')
        spec = {}
        spec.update(spec_encode)
        spec.update(spec_decode)

        print 'spec:'
        print spec.keys()

        self.parameters = ParameterSet(**spec)
        self.parameters.data[:] = np.random.standard_normal(
            self.parameters.data.shape).astype(theano.config.floatX)

    def _init_exprs(self):
        self.exprs = {'inpt': T.tensor3('inpt')}
        P = self.parameters
        
        n_layers = len(self.n_hiddens)
        hidden_to_hiddens = [getattr(P, 'encode_hidden_to_hidden_%i' % i)
                             for i in range(n_layers - 1)]
        recurrents = [getattr(P, 'encode_recurrent_%i' % i)
                      for i in range(n_layers)]
        initial_hiddens = [getattr(P, 'encode_initial_hiddens_%i' % i)
                           for i in range(n_layers)]
        hidden_biases = [getattr(P, 'encode_hidden_bias_%i' % i)
                         for i in range(n_layers)]

        print 'encode'
        print 'hidden_to_hiddens: ', hidden_to_hiddens
        print 'recurrents: ', recurrents
        print 'initial_hiddens: ', initial_hiddens
        print 'hidden_biases: ', hidden_biases
        print
        
        self.exprs.update(rnn.exprs(
            self.exprs['inpt'], P.encode_in_to_hidden, hidden_to_hiddens,
            P.encode_hidden_to_out, hidden_biases, initial_hiddens,
            recurrents, P.encode_out_bias, self.hidden_transfers, self.code_transfer, prefix='encode_'))

        hidden_to_hiddens = [getattr(P, 'decode_hidden_to_hidden_%i' % i)
                             for i in range(n_layers - 1)]
        recurrents = [getattr(P, 'decode_recurrent_%i' % i)
                      for i in range(n_layers)]
        initial_hiddens = [getattr(P, 'decode_initial_hiddens_%i' % i)
                           for i in range(n_layers)]
        hidden_biases = [getattr(P, 'decode_hidden_bias_%i' % i)
                         for i in range(n_layers)]

        print 'decode'
        print 'hidden_to_hiddens: ', hidden_to_hiddens
        print 'recurrents: ', recurrents
        print 'initial_hiddens: ', initial_hiddens
        print 'hidden_biases: ', hidden_biases
        print

        self.exprs.update(rnn.exprs(
            self.exprs['encode_output'], P.decode_in_to_hidden, hidden_to_hiddens,
            P.decode_hidden_to_out, hidden_biases, initial_hiddens,
            recurrents, P.decode_out_bias, self.hidden_transfers, self.out_transfer, prefix='decode_'))

        # supervised stuff
        if self.imp_weight:
            self.exprs['imp_weight'] = T.tensor3('imp_weight')

        print 'all expres: '
        print self.exprs.keys()
        print

        imp_weight = False if not self.imp_weight else self.exprs['imp_weight']
        self.exprs.update(supervised_loss(
            self.exprs['inpt'], self.exprs['decode_output'], self.loss, 2,
            imp_weight=imp_weight))

    def _make_loss_functions(self, mode=None, imp_weight=False):
        """Return pair `f_loss, f_d_loss` of functions.

         - f_loss returns the current loss,
         - f_d_loss returns the gradient of that loss wrt parameters,
           matrix of the loss.
        """
        d_loss = self._d_loss()
        # if self.gradient_clip:
        #     d_loss = project_into_l2_ball(d_loss, self.gradient_clip)

        args = list(self.data_arguments)
        if imp_weight:
            args += ['imp_weight']

        print 'imp_weight=', imp_weight, 'args: ', args
        f_loss = self.function(args, 'loss', explicit_pars=True, mode=mode)
        f_d_loss = self.function(args, d_loss, explicit_pars=True, mode=mode)
        return f_loss, f_d_loss


