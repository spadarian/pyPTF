import numpy as np

from gp_ptf.ptf import PTF


def mahalanobis(data, centroids, W):
    dist = []
    for c in centroids:
        c_rep = np.repeat([c], data.shape[0], 0)
        Dc = c_rep - data
        dist.append((np.matmul(Dc, W) * Dc).sum(1))
    return np.transpose(dist)


def euclidean(data, centroids):
    return np.sqrt(np.power([data - np.repeat([c], data.shape[0], 0) for c in centroids], 2).sum(2)).T


def calc_dist(data, centroids, W, disttype):
    if (disttype == 1):  # Euclidean
        dist = euclidean(data, centroids)
    else:  # Diagonal and Mahalanobis
        dist = mahalanobis(data, centroids, W)
    return dist


def update_centroids(uphi, uephi, a1, dist, data):
    ufi = uphi - (np.full(uphi.shape, a1) * (dist ** (-4)) * np.repeat([uephi], dist.shape[1], 0).T)
    c1 = np.matmul(ufi.T, data)
    t1 = ufi.sum(0)
    t1 = np.repeat([t1], data.shape[1], 0).T
    centroids = c1 / t1
    return centroids


def calc_membership(dist, phi, a1):
    nclass = dist.shape[1]
    tmp = dist ** (-2 / (phi - 1))
    tm2 = dist ** (-2)
    s2 = (a1 * tm2.sum(1)) ** (-1 / (phi - 1))

    t1 = tmp.sum(1)
    t2 = np.repeat([t1], nclass, 0).T + np.repeat([s2], nclass, 0).T
    U = tmp / t2
    Ue = 1 - np.round(U.sum(1), 15)
    return U, Ue


def fuzzy_extragrades(data, alpha, phi, nclass, disttype, maxiter=300,
                      toldif=0.001, exp_eg=None, optim=False):
    if exp_eg is None:
        exp_eg = 1 / (1 + nclass)
    Ue_req = exp_eg
    ndata = data.shape[0]
    ndim = data.shape[1]
    centroids = np.zeros((nclass, ndim))
    dist = np.zeros((ndata, nclass))

    U = np.random.normal(0.5, 0.01, (ndata, nclass))
    U = U / U.sum(1).reshape(-1, 1)

    if (disttype == 1):
        W = np.identity(ndim)
        dist_name = 'Euclidean'
    elif (disttype == 2):
        W = np.identity(ndim) * np.cov(data, rowvar=False)
        dist_name = 'Diagonal'
    elif (disttype == 3):
        W = np.linalg.inv(np.cov(data, rowvar=False))
        dist_name = 'Mahalanobis'

    obj = 0

    Ue = 1 - np.round(U.sum(1), 15)
    uphi = U ** phi
    uephi = Ue ** phi
    a1 = (1 - alpha) / alpha

    # Initialice

    c1 = np.matmul(uphi.T, data)
    t1 = uphi.sum(0)
    t1 = np.repeat([t1], ndim, 0).T
    centroids = c1 / t1

    dist = calc_dist(data, centroids, W, disttype)

    for i in range(1, maxiter + 1):
        # Calculate centroids
        centroids = update_centroids(uphi, uephi, a1, dist, data)

        dist = calc_dist(data, centroids, W, disttype)

        # Save previous iteration
        U_old = U.copy()
        obj_old = obj

        # Calculate new membership matrix
        U, Ue = calc_membership(dist, phi, a1)
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
        if (dif < toldif) and (Udif < toldif):
            break

    # Update centroids
    centroids = update_centroids(uphi, uephi, a1, dist, data)

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
    def __init__(self, nclass, phi, disttype, exp_eg=None, maxiter=300,
                 toldif=0.001):
        self.phi = phi
        self.nclass = nclass
        self.disttype = disttype
        self.exp_eg = exp_eg
        self.maxiter = maxiter
        self.toldif = toldif
        self.centroids = None
        self.U = None
        self.W = None
        self.alpha = None
        self.fitted = False

    def fit(self, data, min_alpha=None, **kwargs):
        from gp_ptf.optim import optim_alpha
        if isinstance(data, PTF):
            try:
                data = data.cleaned_data[data.xs].values
            except Exception:
                raise Exception('There was a problem trying to use the provided PTF')
        alpha = optim_alpha(data,
                            self.phi,
                            self.nclass,
                            self.disttype,
                            min_alpha=min_alpha,
                            **kwargs)
        U, centroids, W = fuzzy_extragrades(data, alpha, self.phi,
                                            self.nclass, self.disttype,
                                            optim=False, **kwargs)
        self.centroids = centroids
        self.U = U
        self.W = W
        self.alpha = alpha
        self.fitted = True

    def dist(self, data):
        if self.fitted:
            return calc_dist(data, self.centroids, self.W, self.disttype)

    def membership(self, data):
        if self.fitted:
            dist = self.dist(data)
            a1 = (1 - self.alpha) / self.alpha
            U, Ue = calc_membership(dist, self.phi, a1)
            return np.concatenate([U, Ue.reshape(-1, 1)], 1)

    @property
    def hard_clusters(self):
        if self.fitted:
            ks = self.U.argmax(1) + 1
            eg = ks.max()
            ks = ks.astype('str')
            ks[ks == str(eg)] = 'Eg'
            return ks

    def PIC(self, y_true, y_pred, conf=0.95):
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
        perc = self.PIC(y_true, y_pred, conf)
        PI = np.matmul(self.U, perc)
        PL = PI + y_pred.reshape(-1, 1)
        return PL

    def PICP(self, y_true, y_pred, conf=0.95):
        PL = self.prediction_limits(y_true, y_pred, conf)
        PICP_count = np.bitwise_and(y_true >= PL[:, 0], y_true <= PL[:, 1])
        PICP = 100 * PICP_count.sum() / len(y_true)
        return PICP

    def MPI(self, y_true, y_pred, conf=0.95):
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
