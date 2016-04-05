import numpy as np
import theano
from theano import tensor as T

from breze.learn.rnn import BaseRnn
from breze.arch.util import ParameterSet, Model
from breze.arch.model.rnn import rnn
from breze.learn.base import UnsupervisedBrezeWrapperBase
from breze.arch.component.common import supervised_loss


class RnnAE(Model, UnsupervisedBrezeWrapperBase):

    def __init__(self, n_inpt, n_hiddens_recog, n_latent, n_hiddens_gen,
                 hidden_recog_transfers, hidden_gen_transfers,
                 latent_transfer='identity',
                 out_transfer='identity',
                 loss='squared',
                 batch_size=None,
                 optimizer='rprop',
                 imp_weight=False,
                 max_iter=1000,
                 verbose=False):

        self.n_inpt = n_inpt
        self.n_hiddens_recog = n_hiddens_recog
        self.n_latent = n_latent
        self.n_hiddens_gen = n_hiddens_gen
        self.hidden_recog_transfers = hidden_recog_transfers
        self.latent_transfer = latent_transfer
        self.hidden_gen_transfers = hidden_gen_transfers
        self.out_transfer = out_transfer
        self.loss = loss
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.imp_weight = imp_weight
        self.max_iter = max_iter
        self.verbose = verbose

        super(RnnAE, self).__init__()

    def _init_pars(self):
        spec = rnn.parameters(self.n_inpt, self.n_hiddens_recog, self.n_latent, prefix='encode_')
        spec.update(rnn.parameters(
            self.n_latent, self.n_hiddens_gen, self.n_inpt, prefix='decode_'))

        self.parameters = ParameterSet(**spec)
        self.parameters.data[:] = np.random.standard_normal(
            self.parameters.data.shape).astype(theano.config.floatX)

    def _init_exprs(self):
        self.exprs = {'inpt': T.tensor3('inpt')}
        P = self.parameters
        
        n_layers = len(self.n_hiddens_recog)
        hidden_to_hiddens = [getattr(P, 'encode_hidden_to_hidden_%i' % i)
                             for i in range(n_layers - 1)]
        recurrents = [getattr(P, 'encode_recurrent_%i' % i)
                      for i in range(n_layers)]
        initial_hiddens = [getattr(P, 'encode_initial_hiddens_%i' % i)
                           for i in range(n_layers)]
        hidden_biases = [getattr(P, 'encode_hidden_bias_%i' % i)
                         for i in range(n_layers)]
        
        self.exprs.update(rnn.exprs(
            self.exprs['inpt'], P.encode_in_to_hidden, hidden_to_hiddens,
            P.encode_hidden_to_out, hidden_biases, initial_hiddens,
            recurrents, P.encode_out_bias, self.hidden_recog_transfers, self.latent_transfer, prefix='encode_'))

        hidden_to_hiddens = [getattr(P, 'decode_hidden_to_hidden_%i' % i)
                             for i in range(n_layers - 1)]
        recurrents = [getattr(P, 'decode_recurrent_%i' % i)
                      for i in range(n_layers)]
        initial_hiddens = [getattr(P, 'decode_initial_hiddens_%i' % i)
                           for i in range(n_layers)]
        hidden_biases = [getattr(P, 'decode_hidden_bias_%i' % i)
                         for i in range(n_layers)]

        self.exprs.update(rnn.exprs(
            self.exprs['encode_output'], P.decode_in_to_hidden, hidden_to_hiddens,
            P.decode_hidden_to_out, hidden_biases, initial_hiddens,
            recurrents, P.decode_out_bias, self.hidden_gen_transfers, self.out_transfer, prefix='decode_'))

        # supervised stuff
        if self.imp_weight:
            self.exprs['imp_weight'] = T.tensor3('imp_weight')

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

        f_loss = self.function(args, 'loss', explicit_pars=True, mode=mode)
        f_d_loss = self.function(args, d_loss, explicit_pars=True, mode=mode)
        return f_loss, f_d_loss


