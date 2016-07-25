# -*- coding: utf-8 -*-


import itertools

from theano.tensor.nnet import batch_normalization
from breze.arch.util import ParameterSet


def invalid_declare(*args, **kwargs):
    raise ValueError('declare cannot be called anymore since '
                     'Layer ws pickled or copied.')


class Layer(object):

    _counter = itertools.count()

    def __init__(self, declare=None, name=None):
        self.make_name(name)

        if declare is None:
            self.parameters = ParameterSet()
            self.declare = self.parameters.declare
        else:
            self.declare = declare

        self._forward()

    def make_name(self, name):
        """Give the layer a unique name.

        If ``name`` is None, construct a name of the form 'N-#' where N is the
        class name and # is a global counter to avoid collisions.
        """
        if name is None:
            self.name = '%s-%i' % (
                self.__class__.__name__, self._counter.next())
        else:
            self.name = name

    def __getstate__(self):
        # The following makes sure that the object can be pickled by replacing
        # the .declare method.
        #
        # Why is it ok to do so? If we pickle a Layer, we can expect it to be
        # already finalized, i.e. _forward has been called. This is being done
        # during initialisation, which means we will not need declare anymore
        # anyway.
        state = self.__dict__.copy()
        state['declare'] = invalid_declare
        return state


class BatchNormalization(object):
    axis = -2

    def __init__(self, inpt, n_inpt, declare, mode='high_mem'):
        self.inpt = inpt
        self.n_inpt = n_inpt

        scale = declare(n_inpt)
        shift = declare(n_inpt)
        mean = inpt.mean(axis=self.axis, keepdims=True)
        std = inpt.std(axis=self.axis, keepdims=True) + 1e-5
        self.output = batch_normalization(inpt, scale, shift, mean, std, mode)


class LayerNormalization(BatchNormalization):
    axis = -1


def normalize(inpt, n_inpt, declare, kind=None):
    if kind is None:
        return inpt

    if kind == 'batch':
        bn = BatchNormalization(inpt, n_inpt, declare)
    elif kind == 'layer':
        bn = LayerNormalization(inpt, n_inpt, declare)
    else:
        raise ValueError('Invalid normalization type: {}'.format(kind))
    return bn.output