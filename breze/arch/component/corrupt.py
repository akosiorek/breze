"""This module contains functionality to corrupt Theano variables
with noise."""

import theano
import theano.tensor as T


# TODO add saltnpepper


def gaussian_perturb(arr, std, rng=None):
    """Return a Theano variable which is perturbed by additive zero-centred
    Gaussian noise with standard deviation ``std``.

    Parameters
    ----------

    arr : Theano variable
        Array of some shape ``n``.

    std : float or scalar Theano variable
        Standard deviation of the Gaussian noise.

    rng : Theano random number generator, optional [default: None]
        Generator to draw random numbers from. If None, rng will be
        instantiated on the spot.


    Returns
    -------

    res : Theano variable
        Of shape ``n``.
    """

    if rng is None:
        rng = T.shared_randomstreams.RandomStreams()
    noise = rng.normal(size=arr.shape, std=std)
    noise = T.cast(noise, theano.config.floatX)
    return arr + noise


def mask(arr, p, rng=None):
    """Return a Theano variable which is with elements of it set to zero with
    probability ``p``.

    Parameters
    ----------

    arr : Theano variable
        Array of some shape ``n``.

    p : float or scalar Theano variable
        Probability that a unit is set to zero.

    rng : Theano random number generator, optional [default: None]
        Generator to draw random numbers from. If None, rng will be
        instantiated on the spot.

    Returns
    -------

    res : Theano variable
        Of shape ``n``.
    """
    if rng is None:
        rng = T.shared_randomstreams.RandomStreams()
    this_mask = T.cast(rng.binomial(size=arr.shape, p=p), theano.config.floatX)
    return arr * this_mask
