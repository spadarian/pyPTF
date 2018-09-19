import numpy as np
from scipy.spatial.distance import mahalanobis, euclidean


def calc_dist(data, centroids, W, disttype):
    if (disttype == 1):  # Euclidean
        dist = np.sqrt(np.array([[euclidean(r, c) for c in centroids] for r in data]))
    else:  # Diagonal and Mahalanobis
        dist = np.sqrt(np.array([[mahalanobis(r, c, W) for c in centroids] for r in data]))
    return dist


def update_centroids(uphi, uephi, a1, dist, data):
    ufi = uphi - (np.full(uphi.shape, a1) * (dist ** (-4)) * np.repeat([uephi], dist.shape[1], 0).T)
    c1 = np.matmul(ufi.T, data)
    t1 = ufi.sum(0)
    t1 = np.repeat([t1], data.shape[1], 0).T
    centroids = c1 / t1
    return centroids


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
        tmp = dist ** (-2 / (phi - 1))
        tm2 = dist ** (-2)
        s2 = (a1 * tm2.sum(1)) ** (-1 / (phi - 1))

        t1 = tmp.sum(1)
        t2 = np.repeat([t1], nclass, 0).T + np.repeat([s2], nclass, 0).T
        U = tmp / t2
        Ue = 1 - np.round(U.sum(1), 15)
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

    def conf(x):
        x_sorted = sorted(x, reverse=True)
        return x_sorted[1] / x_sorted[0]

    CI = np.apply_along_axis(conf, 1, U_end)

    if optim:
        return np.abs(Ue_mean - Ue_req)
    return [U_end, centroids, W]
