import logging
import multiprocessing

import numpy as np
import pygmo as pg

from gp_ptf.fkmeans import fuzzy_extragrades

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
        return [fuzzy_extragrades(**self.params)]

    def get_bounds(self):
        return ([self.min_alpha], [1])


def optim_alpha(data, phi, nclass, disttype, algo=None, archi=None,
                verbose=1, min_alpha=None, **kwargs):
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
