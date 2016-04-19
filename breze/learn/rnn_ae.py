import numpy as np
import theano
from theano import tensor as T

from breze.arch.util import ParameterSet, Model, get_named_variables
from breze.arch.model.rnn import rnn, lstm
from breze.learn.base import SupervisedBrezeWrapperBase, UnsupervisedBrezeWrapperBase
from breze.arch.component.common import supervised_loss
from breze.arch.component import corrupt


class GenericRnnAE(Model):

    encode_name = 'recog'
    decode_name = 'gen'
    sample_dim = 1,

    def __init__(self, n_inpt, n_hiddens_recog, n_latent, n_hiddens_gen,
                 hidden_recog_transfers, hidden_gen_transfers,
                 latent_transfer='identity',
                 out_transfer='identity',
                 loss='squared',
                 tied_weights=False,
                 batch_size=None,
                 optimizer='rprop',
                 imp_weight=False,
                 max_iter=1000,
                 gradient_clip=False,
                 verbose=False,
                 skip_to_out=False):

        self.n_inpt = n_inpt
        self.n_hiddens_recog = n_hiddens_recog
        self.n_latent = n_latent
        self.n_hiddens_gen = n_hiddens_gen
        self.hidden_recog_transfers = hidden_recog_transfers
        self.latent_transfer = latent_transfer
        self.hidden_gen_transfers = hidden_gen_transfers
        self.out_transfer = out_transfer
        self.loss = loss
        self.tied_weights = tied_weights
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.imp_weight = imp_weight
        self.max_iter = max_iter
        self.gradient_clip = gradient_clip
        self.verbose = verbose
        self.skip_to_out = skip_to_out

        super(GenericRnnAE, self).__init__()

    def _make_spec(self):

        spec = rnn.parameters(self.n_inpt, self.n_hiddens_recog, self.n_latent, self.skip_to_out, prefix=self.encode_name + '_')
        if not self.tied_weights:
            spec.update(rnn.parameters(
                self.n_latent, self.n_hiddens_gen, self.n_inpt, prefix=self.decode_name + '_')
            )
        else:
            spec.update({'{}_out_bias'.format(self.decode_name): self.n_inpt})

            assert self.n_hiddens_recog == self.n_hiddens_gen[::-1], 'Layers do not match for tied weights'
            assert self.hidden_recog_transfers == self.hidden_gen_transfers[::-1]

        return spec

    def _make_exprs(self, name, inpt_expr_name, out_transfer, reverse=False):

        p = self.parameters
        inpt_expr = self.exprs[inpt_expr_name]

        true_name = name
        if reverse:
            true_name = name
            name = reverse

        n_layers = len(getattr(self, 'n_hiddens_{}'.format(name)))
        hidden_to_hiddens = [getattr(p, '{}_hidden_to_hidden_{}'.format(name, i))
                             for i in range(n_layers - 1)]
        recurrents = [getattr(p, '{}_recurrent_{}'.format(name, i))
                      for i in range(n_layers)]
        initial_hiddens = [getattr(p, '{}_initial_hiddens_{}'.format(name, i))
                           for i in range(n_layers)]
        hidden_biases = [getattr(p, '{}_hidden_bias_{}'.format(name, i))
                         for i in range(n_layers)]

        hidden_transfers = getattr(self, 'hidden_{}_transfers'.format(name))

        in_to_hidden = getattr(p, '{}_in_to_hidden'.format(name))
        hidden_to_out = getattr(p, '{}_hidden_to_out'.format(name))
        out_bias = getattr(p, '{}_out_bias'.format(true_name))

        if self.skip_to_out:
            in_to_out = getattr(p, '{}_in_to_out'.format(name))
        else:
            in_to_out = None

        if reverse:
            hidden_to_hiddens = [w.T for w in reversed(hidden_to_hiddens)]
            recurrents, initial_hiddens, hidden_biases, hidden_transfers = [
                l[::-1] for l in (recurrents, initial_hiddens, hidden_biases, hidden_transfers)]

            in_to_hidden, hidden_to_out = hidden_to_out.T, in_to_hidden.T

            if in_to_out is not None:
                in_to_out = in_to_out.T

        exprs = rnn.exprs(
            inpt_expr, in_to_hidden, hidden_to_hiddens,
            hidden_to_out, hidden_biases, initial_hiddens,
            recurrents, out_bias, hidden_transfers, out_transfer, in_to_out=in_to_out)

        return {'{}_{}'.format(true_name, k): v for k, v in exprs.iteritems()}

    def _init_pars(self):
        spec = self._make_spec()

        self.parameters = ParameterSet(**spec)
        self.parameters.data[:] = np.random.standard_normal(
            self.parameters.data.shape).astype(theano.config.floatX)

    def _init_exprs(self):
        self.exprs = {'inpt': T.tensor3('inpt')}

        reverse = self.encode_name if self.tied_weights else False
        self.exprs.update(self._make_exprs(self.encode_name, 'inpt', self.latent_transfer))
        self.exprs.update(self._make_exprs(
            self.decode_name, self.encode_name + '_output', self.out_transfer, reverse=reverse))

        # loss exprs
        if self.imp_weight:
            self.exprs['imp_weight'] = T.tensor3('imp_weight')

        imp_weight = False if not self.imp_weight else self.exprs['imp_weight']
        self.exprs.update(supervised_loss(
            self.exprs['inpt'], self.exprs[self.decode_name + '_output'], self.loss, 2,
            imp_weight=imp_weight))


class DenoisingMixin(object):

    def __init__(self, noise_type, c_noise):
        self.noise_type = noise_type
        self.c_noise = c_noise

    def _init_exprs(self):

        if self.noise_type == 'gauss':
            corrupted_inpt = corrupt.gaussian_perturb(
                self.exprs['inpt'], self.c_noise)
        elif self.noise_type == 'mask':
            corrupted_inpt = corrupt.mask(
                self.exprs['inpt'], self.c_noise)
                
        corrupted_output_name = GenericRnnAE.decode_name + '_output'
        output_from_corrupt = theano.clone(
            self.exprs[corrupted_output_name],
            {self.exprs['inpt']: corrupted_inpt}
        )

        score = self.exprs['loss']
        loss = theano.clone(
            self.exprs['loss'],
            {self.exprs[corrupted_output_name]: output_from_corrupt})

        self.exprs.update(get_named_variables(locals(), overwrite=True))


class RnnAE(GenericRnnAE, UnsupervisedBrezeWrapperBase): pass


class DenoisingRnnAE(RnnAE, DenoisingMixin):
    def __init__(self, n_inpt, n_hiddens_recog, n_latent, n_hiddens_gen,
                 hidden_recog_transfers, hidden_gen_transfers,
                 latent_transfer='identity',
                 out_transfer='identity',
                 loss='squared',
                 tied_weights=False,
                 batch_size=None,
                 optimizer='rprop',
                 imp_weight=False,
                 max_iter=1000,
                 gradient_clip=False,
                 verbose=False,
                 noise_type='gauss', c_noise=.2):

        DenoisingMixin.__init__(self, noise_type, c_noise)
        RnnAE.__init__(self, n_inpt, n_hiddens_recog, n_latent, n_hiddens_gen,
                       hidden_recog_transfers, hidden_gen_transfers,
                       latent_transfer,
                       out_transfer,
                       loss,
                       tied_weights,
                       batch_size,
                       optimizer,
                       imp_weight,
                       max_iter,
                       gradient_clip,
                       verbose)

    def _init_exprs(self):
        RnnAE._init_exprs(self)
        DenoisingMixin._init_exprs(self)


class LstmAE(Model, UnsupervisedBrezeWrapperBase):

    sample_dim = 1,
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


class LadderRnn(GenericRnnAE, DenoisingMixin, SupervisedBrezeWrapperBase):

    predict_name = 'pred'
    sample_dim = 1, 1

    def __init__(self, n_inpt, n_hiddens_recog, n_latent, n_hiddens_gen,
                 n_hiddens_pred, hidden_recog_transfers, hidden_gen_transfers,
                 hidden_pred_transfers, latent_transfer='identity',
                 out_transfer='identity',
                 loss='squared',
                 tied_weights=False,
                 batch_size=None,
                 optimizer='rprop',
                 imp_weight=False,
                 max_iter=1000,
                 gradient_clip=False,
                 verbose=False,
                 skip_to_out=False,
                 noise_type='gauss', c_noise=.2):

        self.n_hiddens_pred = n_hiddens_pred
        self.hidden_pred_transfers = hidden_pred_transfers

        DenoisingMixin.__init__(self, noise_type, c_noise)
        GenericRnnAE.__init__(self, n_inpt, n_hiddens_recog, n_latent, n_hiddens_gen,
                                        hidden_recog_transfers, hidden_gen_transfers,
                                        latent_transfer,
                                        out_transfer,
                                        loss,
                                        tied_weights,
                                        batch_size,
                                        optimizer,
                                        imp_weight,
                                        max_iter,
                                        gradient_clip,
                                        verbose,
                                        skip_to_out)

    def _make_spec(self):
        spec = super(LadderRnn, self)._make_spec()
        spec.update(rnn.parameters(
            self.n_latent, self.n_hiddens_pred, self.n_inpt, self.skip_to_out, prefix=self.predict_name + '_'))
        return spec

    def _init_exprs(self):
        GenericRnnAE._init_exprs(self)
        DenoisingMixin._init_exprs(self)

        self.exprs['target'] = T.tensor3('target')

        input_name = GenericRnnAE.encode_name + '_output'
        self.exprs.update(self._make_exprs(self.predict_name, input_name, self.out_transfer))
        
        # TODO: figure out how to add loss with next timestep's input as target
        # TODO: enable imp weights; there might be different ones for input and output(?)
        reconstruction_loss = self.exprs['loss']
        prediction_loss = supervised_loss(
            self.exprs['target'], self.exprs[self.predict_name + '_output'], self.loss, 2,
            imp_weight=False)['loss']

        self.exprs['loss'] = reconstruction_loss + prediction_loss
