import numpy as np
import theano
from theano import tensor as T

from breze.learn.rnn import BaseRnn
from breze.arch.util import ParameterSet, Model, get_named_variables
from breze.arch.model.rnn import rnn, lstm
from breze.learn.base import UnsupervisedBrezeWrapperBase
from breze.arch.component.common import supervised_loss
from breze.arch.component.misc import project_into_l2_ball
from breze.arch.component import corrupt


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
                 gradient_clip=False,
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
        self.gradient_clip = gradient_clip
        self.verbose = verbose

        super(RnnAE, self).__init__()

    def _make_spec(self):
        spec = rnn.parameters(self.n_inpt, self.n_hiddens_recog, self.n_latent, prefix='encode_')
        spec.update(rnn.parameters(
            self.n_latent, self.n_hiddens_gen, self.n_inpt, prefix='decode_'))
        return spec

    def _init_pars(self):
        spec = self._make_spec()

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

        exprs = rnn.exprs(
            self.exprs['inpt'], P.encode_in_to_hidden, hidden_to_hiddens,
            P.encode_hidden_to_out, hidden_biases, initial_hiddens,
            recurrents, P.encode_out_bias, self.hidden_recog_transfers, self.latent_transfer)

        self.exprs.update({'encode_{0}'.format(k): v for k, v in exprs.iteritems()})

        hidden_to_hiddens = [getattr(P, 'decode_hidden_to_hidden_%i' % i)
                             for i in range(n_layers - 1)]
        recurrents = [getattr(P, 'decode_recurrent_%i' % i)
                      for i in range(n_layers)]
        initial_hiddens = [getattr(P, 'decode_initial_hiddens_%i' % i)
                           for i in range(n_layers)]
        hidden_biases = [getattr(P, 'decode_hidden_bias_%i' % i)
                         for i in range(n_layers)]

        exprs = rnn.exprs(
            self.exprs['encode_output'], P.decode_in_to_hidden, hidden_to_hiddens,
            P.decode_hidden_to_out, hidden_biases, initial_hiddens,
            recurrents, P.decode_out_bias, self.hidden_gen_transfers, self.out_transfer)
        self.exprs.update({'decode_{0}'.format(k): v for k, v in exprs.iteritems()})

        # supervised stuff
        if self.imp_weight:
            self.exprs['imp_weight'] = T.tensor3('imp_weight')

        imp_weight = False if not self.imp_weight else self.exprs['imp_weight']
        self.exprs.update(supervised_loss(
            self.exprs['inpt'], self.exprs['decode_output'], self.loss, 2,
            imp_weight=imp_weight))


class DenoisingRnnAE(RnnAE):

    def __init__(self, n_inpt, n_hiddens_recog, n_latent, n_hiddens_gen,
                 hidden_recog_transfers, hidden_gen_transfers,
                 latent_transfer='identity',
                 out_transfer='identity',
                 loss='squared',
                 batch_size=None,
                 optimizer='rprop',
                 imp_weight=False,
                 max_iter=1000,
                 gradient_clip=False,
                 verbose=False,
                 noise_type='gauss', c_noise=.2):

        self.noise_type = noise_type
        self.c_noise = c_noise

        super(DenoisingRnnAE, self).__init__(n_inpt, n_hiddens_recog, n_latent, n_hiddens_gen,
                                    hidden_recog_transfers, hidden_gen_transfers,
                                    latent_transfer,
                                    out_transfer,
                                    loss,
                                    batch_size,
                                    optimizer,
                                    imp_weight,
                                    max_iter,
                                    gradient_clip,
                                    verbose)

    def _init_exprs(self):

        super(DenoisingRnnAE, self)._init_exprs()
        if self.noise_type == 'gauss':
            corrupted_inpt = corrupt.gaussian_perturb(
                self.exprs['inpt'], self.c_noise)
        elif self.noise_type == 'mask':
            corrupted_inpt = corrupt.mask(
                self.exprs['inpt'], self.c_noise)

        output_from_corrupt = theano.clone(
            self.exprs['encode_output'],
            {self.exprs['inpt']: corrupted_inpt}
        )

        score = self.exprs['loss']
        loss = theano.clone(
            self.exprs['loss'],
            {self.exprs['encode_output']: output_from_corrupt})

        self.exprs.update(get_named_variables(locals(), overwrite=True))


class LstmAE(Model, UnsupervisedBrezeWrapperBase):

    def __init__(self, n_inpt, n_hiddens_recog, n_latent, n_hiddens_gen,
                 hidden_recog_transfers, hidden_gen_transfers,
                 latent_transfer='identity',
                 out_transfer='identity',
                 loss='squared',
                 batch_size=None,
                 optimizer='rprop',
                 imp_weight=False,
                 max_iter=1000,
                 gradient_clip=None,
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
        self.gradient_clip = gradient_clip
        self.verbose = verbose

        super(LstmAE, self).__init__()

    def _init_pars(self):
        spec_encode = lstm.parameters(self.n_inpt, self.n_hiddens_recog, self.n_latent)
        spec = {'{0}_{1}'.format('encode', k): v for (k, v) in spec_encode.iteritems()}
        spec_decode = lstm.parameters(self.n_latent, self.n_hiddens_gen, self.n_inpt)
        spec.update({'{0}_{1}'.format('decode', k): v for (k, v) in spec_decode.iteritems()})

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
        hidden_biases = [getattr(P, 'encode_hidden_bias_%i' % i)
                             for i in range(n_layers)]

        ingate_peepholes = [getattr(P, 'encode_ingate_peephole_%i' % i)
                             for i in range(n_layers)]
        outgate_peepholes = [getattr(P, 'encode_outgate_peephole_%i' % i)
                             for i in range(n_layers)]
        forgetgate_peepholes = [getattr(P, 'encode_forgetgate_peephole_%i' % i)
                             for i in range(n_layers)]

        exprs = lstm.exprs(
            self.exprs['inpt'], P.encode_in_to_hidden, hidden_to_hiddens,
            P.encode_hidden_to_out, hidden_biases, recurrents, P.encode_out_bias,
             ingate_peepholes, outgate_peepholes, forgetgate_peepholes,
            self.hidden_recog_transfers, self.latent_transfer, pooling=None)

        self.exprs.update({'encode_{0}'.format(k): v for k, v in exprs.iteritems()})

        hidden_to_hiddens = [getattr(P, 'decode_hidden_to_hidden_%i' % i)
                             for i in range(n_layers - 1)]
        recurrents = [getattr(P, 'decode_recurrent_%i' % i)
                             for i in range(n_layers)]
        hidden_biases = [getattr(P, 'decode_hidden_bias_%i' % i)
                             for i in range(n_layers)]
        ingate_peepholes = [getattr(P, 'decode_ingate_peephole_%i' % i)
                             for i in range(n_layers)]
        outgate_peepholes = [getattr(P, 'decode_outgate_peephole_%i' % i)
                             for i in range(n_layers)]
        forgetgate_peepholes = [getattr(P, 'decode_forgetgate_peephole_%i' % i)
                             for i in range(n_layers)]

        exprs = lstm.exprs(
            self.exprs['encode_output'], P.decode_in_to_hidden, hidden_to_hiddens,
            P.decode_hidden_to_out, hidden_biases, recurrents, P.decode_out_bias,
            ingate_peepholes, outgate_peepholes, forgetgate_peepholes,
            self.hidden_gen_transfers, self.out_transfer, pooling=None)

        self.exprs.update({'decode_{0}'.format(k): v for k, v in exprs.iteritems()})

        # supervised stuff
        if self.imp_weight:
            self.exprs['imp_weight'] = T.tensor3('imp_weight')

        imp_weight = False if not self.imp_weight else self.exprs['imp_weight']
        self.exprs.update(supervised_loss(
            self.exprs['inpt'], self.exprs['decode_output'], self.loss, 2,
            imp_weight=imp_weight))


class LadderRnn(DenoisingRnnAE):
    def __init__(self, n_inpt, n_hiddens_recog, n_latent, n_hiddens_gen,
                 hidden_recog_transfers, hidden_gen_transfers,
                 latent_transfer='identity',
                 out_transfer='identity',
                 loss='squared',
                 batch_size=None,
                 optimizer='rprop',
                 imp_weight=False,
                 max_iter=1000,
                 gradient_clip=False,
                 verbose=False,
                 noise_type='gauss', c_noise=.2):

        super(LadderRnn, self).__init__(n_inpt, n_hiddens_recog, n_latent, n_hiddens_gen,
                                        hidden_recog_transfers, hidden_gen_transfers,
                                        latent_transfer,
                                        out_transfer,
                                        loss,
                                        batch_size,
                                        optimizer,
                                        imp_weight,
                                        max_iter,
                                        gradient_clip,
                                        verbose,
                                        noise_type,
                                        c_noise)

    def _init_pars(self):
        spec = self._make_spec()
        spec.update(rnn.parameters(
            self.n_latent, self.n_hiddens_pred, self.n_inpt, prefix='predict_'))

        self.parameters = ParameterSet(**spec)
        self.parameters.data[:] = np.random.standard_normal(
            self.parameters.data.shape).astype(theano.config.floatX)

    def _init_exprs(self):
        super(LadderRnn, self)._init_exprs()

        # TODO: figure out how to add loss with next timestep's input as target
        # TODO: enable imp weights; there might be diferent ones for input and output(?)
        reconstruction_loss = self.expr['loss']
        prediction_loss = supervised_loss(
            self.exprs['target'], self.exprs['predict_output'], self.loss, 2,
            imp_weight=False)
