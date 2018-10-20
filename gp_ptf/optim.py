"""Module with optimisation methods.

This module is not intended to be used by itself. The optimisation can be configured
when calling the :meth:`.fkmeans.FKMEx.fit` method.

"""
import logging
import multiprocessing

import numpy as np
import pygmo as pg

from .fkmeans import _fuzzy_extragrades

logger = logging.getLogger(__name__)


class PTFOptim(object):

    def __init__(self, data, phi, nclass, disttype, min_alpha=None, **kwargs):
        self.params = {
            'data': data,
            'phi': phi,
            'nclass': nclass,
            'disttype': disttype
        }
        self.params.update(kwargs)
        if not min_alpha:
            min_alpha = 0.0000001
        logger.info('Using min_alpha: {}'.format(min_alpha))
        self.min_alpha = min_alpha

    def fitness(self, x):
        kwargs = self.params
        kwargs['alpha'] = x
        kwargs['optim'] = True
        return [_fuzzy_extragrades(**self.params)]

    def get_bounds(self):
        return ([self.min_alpha], [1])


def optim_alpha(data, phi, nclass, disttype, algo=None, archi=None,
                verbose=1, min_alpha=None, **kwargs):
    """Find optim alpha.

    Parameters
    ----------
    data : np.ndarray
        Data to cluster.
    phi : float
        Degree of fuzziness or overlap of the generated clusters.
        Usually in the range [1, 2].
    nclass : int
        Number of cluster to generate.
    disttype : {'euclidean', 'diagonal', 'mahalanobis'}
            Type of distance (the default is 'mahalanobis').
    algo : pygmo.algorithm
        See `pygmo Algorithm documentation <https://esa.github.io/pagmo2/docs/python/py_algorithm.html>`_.
    archi : pygmo.archipelago
        See `pygmo Archipelago documentation <https://esa.github.io/pagmo2/docs/python/py_archipelago.html>`_.
    verbose : int, bool
        Show optimisation feedback (the default is 1).
        Currently not used.
    min_alpha : float
        The optimisation finds the optimal alpha value in the range [`min_alpha`, 1].
    **kwargs :
        Parameters passed to the :func:`~.fkmeans._fuzzy_extragrades` function.

    Returns
    -------
    float
        Optimum alpha value.

    """
    prob = pg.problem(PTFOptim(data, phi, nclass, disttype, min_alpha, **kwargs))
    if not algo:
        algo = pg.algorithm(pg.sade(gen=20, ftol=0.0005))
    if not archi:
        logger.info('Initialising polulation')
        archi = pg.archipelago(n=multiprocessing.cpu_count(),
                               algo=algo,
                               prob=prob,
                               pop_size=7)
    logger.info('Starting evolution...')
    archi.evolve()
    archi.wait()
    best = np.argmin(archi.get_champions_f())
    logger.info('Done!')
    best_alpha = archi.get_champions_x()[best][0]
    logger.info('Optim alpha: {}'.format(best_alpha))
    return best_alpha
