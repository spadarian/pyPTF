"""Module to perform uncertainty assessment."""

import logging

import numpy as np

from .ptf import PTF

logger = logging.getLogger(__name__)


def mahalanobis(data, centroids, W):
    """Calculate the mahalanobis distance between observations and centroids.

    Parameters
    ----------
    data : np.ndarray
        2-dimensional array containing the observations.
    centroids : np.ndarray
        2-dimensional `centroids` array. It should have the same number of columns
        than `data`.
    W : np.ndarray
        The inverse of the covariance matrix.

    Returns
    -------
    np.ndarray
        Distance matrix.

    """
    dist = []
    for c in centroids:
        c_rep = np.repeat([c], data.shape[0], 0)
        Dc = c_rep - data
        dist.append((np.matmul(Dc, W) * Dc).sum(1))
    return np.transpose(dist)


def euclidean(data, centroids):
    """Calculate the euclidean distance between observations and centroids.

    Parameters
    ----------
    data : np.ndarray
        2-dimensional array containing the observations.
    centroids : np.ndarray
        2-dimensional `centroids` array. It should have the same number of columns
        than `data`.

    Returns
    -------
    np.ndarray
        Distance matrix.

    """
    return np.sqrt(np.power([data - np.repeat([c], data.shape[0], 0) for c in centroids], 2).sum(2)).T


def _calc_dist(data, centroids, W, disttype):
    if (disttype == 'euclidean'):
        dist = euclidean(data, centroids)
    else:  # Diagonal and Mahalanobis
        dist = mahalanobis(data, centroids, W)
    return dist


def _update_centroids(uphi, uephi, a1, dist, data):
    ufi = uphi - (np.full(uphi.shape, a1) * (dist ** (-4)) * np.repeat([uephi], dist.shape[1], 0).T)
    c1 = np.matmul(ufi.T, data)
    t1 = ufi.sum(0)
    t1 = np.repeat([t1], data.shape[1], 0).T
    centroids = c1 / t1
    return centroids


def _calc_membership(dist, phi, a1):
    nclass = dist.shape[1]
    tmp = dist ** (-2 / (phi - 1))
    tm2 = dist ** (-2)
    s2 = (a1 * tm2.sum(1)) ** (-1 / (phi - 1))

    t1 = tmp.sum(1)
    t2 = np.repeat([t1], nclass, 0).T + np.repeat([s2], nclass, 0).T
    U = tmp / t2
    Ue = 1 - U.sum(1)
    return np.abs(U), np.abs(Ue)


def _fuzzy_extragrades(data, alpha, phi, nclass, disttype, maxiter=300,
                       toldiff=0.001, exp_eg=None, optim=False):
    """Run clustering routine.

    This functions in not supposed to be used by itself. It is internally used
    by :class:`.FKMEx`.

    See :class:`.FKMEx` for a complete list of parameters.

    Parameters
    ----------
    optim : bool
        Whether to run in alpha optimisation mode or clustering mode.

    """
    if exp_eg is None:
        exp_eg = 1 / (1 + nclass)
    Ue_req = exp_eg
    ndata = data.shape[0]
    ndim = data.shape[1]
    centroids = np.zeros((nclass, ndim))
    dist = np.zeros((ndata, nclass))

    U = np.random.normal(0.5, 0.01, (ndata, nclass))
    U = np.abs(U / U.sum(1).reshape(-1, 1))

    if (disttype == 'euclidean'):
        W = np.identity(ndim)
    elif (disttype == 'diagonal'):
        W = np.identity(ndim) * np.cov(data, rowvar=False)
    elif (disttype == 'mahalanobis'):
        W = np.linalg.inv(np.cov(data, rowvar=False))

    obj = 0

    Ue = np.abs(1 - U.sum(1))
    uphi = U ** phi
    uephi = Ue ** phi
    a1 = (1 - alpha) / alpha

    # Initialice

    c1 = np.matmul(uphi.T, data)
    t1 = uphi.sum(0)
    t1 = np.repeat([t1], ndim, 0).T
    centroids = c1 / t1

    dist = _calc_dist(data, centroids, W, disttype)

    for i in range(1, maxiter + 1):
        # Calculate centroids
        centroids = _update_centroids(uphi, uephi, a1, dist, data)

        dist = _calc_dist(data, centroids, W, disttype)

        # Save previous iteration
        U_old = U.copy()
        obj_old = obj

        # Calculate new membership matrix
        U, Ue = _calc_membership(dist, phi, a1)
        uphi = U ** phi
        uephi = Ue ** phi

        # Calculate obj function
        o1 = (dist ** 2) * uphi
        d2 = dist ** (-2)
        o2 = uephi * d2.sum(1)
        obj = alpha * o1.sum() + (1 - alpha) * o2.sum()

        # Check for convergence
        dif = obj_old - obj
#         difU = np.sqrt((U - U_old) * (U - U_old))
        difU = np.abs(U - U_old)
        Udif = difU.sum()
        if (dif < toldiff) and (Udif < toldiff):
            break

    # Update centroids
    centroids = _update_centroids(uphi, uephi, a1, dist, data)

    U_end = np.concatenate([U, Ue.reshape(-1, 1)], 1)

    hard_clust = (U_end.argmax(1) + 1).astype(str)
    is_eg = hard_clust == str(U_end.shape[1])
    hard_clust[is_eg] = 'Eg'

    Ue_mean = Ue.mean()

    #   dist_centroids = mahaldist(centroids)

    # def conf(x):
    #     x_sorted = sorted(x, reverse=True)
    #     return x_sorted[1] / x_sorted[0]
    #
    # CI = np.apply_along_axis(conf, 1, U_end)

    if optim:
        return np.abs(Ue_mean - Ue_req)
    return [U_end, centroids, W]


class FKMEx(object):
    """Fuzzy k-means with extragrades.

    Parameters
    ----------
    nclass : int
        Number of cluster to generate.
    phi : float
        Degree of fuzziness or overlap of the generated clusters.
        Usually in the range [1, 2].
    disttype : {'euclidean', 'diagonal', 'mahalanobis'}
        Type of distance (the default is 'mahalanobis').
    exp_eg : float
        The expected fraction of extragrades. If not provided, it is assumed to
        be dependant on the number of clusters (1 / (1 + nclass)).
    maxiter : int
        Maximum number of iterations (the default is 300).
    toldiff : float
        Minimum difference to converge (the default is 0.001).
    alpha : float
        Mean membership of the extragrade class (the default is None).
        This parameter is usually obtained by optimisation. See :func:`FKMEx.fit`.
        Once obtained by optimisation, it can be provided to skip the optimisation.

    Attributes
    ----------
    centroids : np.ndarray
        Coordinates of the centroids.
    U : np.ndarray
        Memebership matrix with shape (number_obs, nclass + 1).
        The last com
    W : np.ndarray
        The inverse of the data covariance matrix.
    fitted : bool
        True if the :func:`FKMEx.fit` method has been called.
    phi
    nclass
    disttype
    exp_eg
    maxiter
    toldiff
    alpha

    """
    def __init__(self, nclass, phi, disttype='mahalanobis', exp_eg=None,
                 maxiter=300, toldiff=0.001, alpha=None):
        self.phi = phi
        self.nclass = nclass
        self.disttype = disttype
        self.exp_eg = exp_eg
        self.maxiter = maxiter
        self.toldiff = toldiff
        self.centroids = None
        self.U = None
        self.W = None
        self.alpha = alpha
        self.fitted = False

    def fit(self, data, min_alpha=None, **kwargs):
        """Run the fuzzy k-means with extragrades algorithm.

        Parameters
        ----------
        data : np.ndarray or :class:`~.ptf.PTF`
            Data to cluster.
        min_alpha : float
            The optimisation finds the optimal alpha value in the range [`min_alpha`, 1].
            If None (the default) the search range is [0, 1].
            When a series of consecutive number of clusters are run (i.e.: nclass=2; nclass=3, etc.),
            the first FKMEx should search in the range [0, 1] and the next one in the range
            [min_alpha=optim_alpha_previous, 1].
        **kwargs :
            Parameters passed to the :func:`~.optim.optim_alpha` function.

        Returns
        -------
        FKMEx
            Fitted self.

        """
        if isinstance(data, PTF):
            try:
                data = data.cleaned_data[data.xs].values
            except Exception:
                raise Exception('There was a problem trying to use the provided PTF')
        if self.alpha is None:
            from .optim import optim_alpha
            alpha = optim_alpha(data,
                                self.phi,
                                self.nclass,
                                self.disttype,
                                min_alpha=min_alpha,
                                **kwargs)
        else:
            alpha = self.alpha
            logger.info('Skipping alpha optimisation')
            logger.info('Using provided alpha: {}'.format(alpha))
        U, centroids, W = _fuzzy_extragrades(data, alpha, self.phi,
                                             self.nclass, self.disttype,
                                             optim=False, **kwargs)
        self.centroids = centroids
        self.U = U
        self.W = W
        self.alpha = alpha
        self.fitted = True
        return self

    def dist(self, data):
        """Calculate the distance between data and the class centroids.

        Parameters
        ----------
        data : np.ndarray
            2-dimensional array containing the observations.

        Returns
        -------
        np.ndarray
            Distance matrix.

        """
        if self.fitted:
            return _calc_dist(data, self.centroids, self.W, self.disttype)

    def membership(self, data):
        """Calculate the membership to each class.

        Parameters
        ----------
        data : np.ndarray
            2-dimensional array containing the observations.

        Returns
        -------
        np.ndarray
            Memebership matrix.

        """
        if self.fitted:
            dist = self.dist(data)
            a1 = (1 - self.alpha) / self.alpha
            U, Ue = _calc_membership(dist, self.phi, a1)
            return np.concatenate([U, Ue.reshape(-1, 1)], 1)

    @property
    def hard_clusters(self):
        """Defuzzified classes."""
        if self.fitted:
            ks = self.U.argmax(1) + 1
            eg = ks.max()
            ks = ks.astype('str')
            ks[ks == str(eg)] = 'Eg'
            return ks

    def _PIC(self, y_true, y_pred, conf=0.95):
        hard = self.hard_clusters
        ks = np.unique(hard)
        alpha = 100 - conf * 100
        perc = []
        for k in ks:
            logic = hard == k
            k_pred = y_pred[logic]
            k_obs = y_true[logic]
            res = k_obs - k_pred
            perc.append(np.percentile(res, [alpha / 2, 100 - (alpha / 2)]))
        perc = np.array(perc)
        perc[-1] *= 2
        return perc

    def prediction_limits(self, y_true, y_pred, conf=0.95):
        """Calculate prediction limits.

        Parameters
        ----------
        y_true : np.ndarray
            Observed values of the target variable.
        y_pred : np.ndarray
            Predicted values of the target variable.
        conf : float
            Confidence level used when estimimating the prediction_interval (the default is 0.95).

        Returns
        -------
        np.ndarray
            Prediction limits (`y_pred` ± uncertainty).

        """
        perc = self._PIC(y_true, y_pred, conf)
        PI = np.matmul(self.U, perc)
        PL = PI + y_pred.reshape(-1, 1)
        return PL

    def PICP(self, y_true, y_pred, conf=0.95):
        """Calculate prediction interval coverage probability.

        Parameters
        ----------
        y_true : np.ndarray
            Observed values of the target variable.
        y_pred : np.ndarray
            Predicted values of the target variable.
        conf : float
            Confidence level used when estimimating the prediction_interval (the default is 0.95).

        Returns
        -------
        float
            Prediction interval coverage probability.

        """
        PL = self.prediction_limits(y_true, y_pred, conf)
        PICP_count = np.bitwise_and(y_true >= PL[:, 0], y_true <= PL[:, 1])
        PICP = 100 * PICP_count.sum() / len(y_true)
        return PICP

    def MPI(self, y_true, y_pred, conf=0.95):
        """Calcululate mean prediction interval.

        Parameters
        ----------
        y_true : np.ndarray
            Observed values of the target variable.
        y_pred : np.ndarray
            Predicted values of the target variable.
        conf : float
            Confidence level used when estimimating the prediction_interval (the default is 0.95).

        Returns
        -------
        float
            Mean prediction interval.

        """
        PL = self.prediction_limits(y_true, y_pred, conf)
        MPI = (PL[:, 1] - PL[:, 0]).sum() / len(y_true)
        return MPI

    def __repr__(self):
        main_str = 'classes={} phi={} dist={}'.format(self.nclass,
                                                      self.phi,
                                                      self.disttype)
        if self.fitted:
            fitted = '(fitted) '
        else:
            fitted = ''
        rep = '<{} {}{}>'.format(self.__class__.__name__,
                                 fitted,
                                 main_str)
        return rep
