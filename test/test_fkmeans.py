import pytest
import numpy as np

from pyPTF.fkmeans import _calc_dist, _calc_membership, _update_centroids, FKMEx

d = np.array([[0, 2], [1, 1], [2.5, 0]])
dist_euc = np.array([
    [1, 3.60555128],
    [1, 3.60555128],
    [2.5, 4.03112887]
])
dist_mah = np.array([
    [48., 1456.],
    [76., 1596.],
    [52., 1468.]
])

alpha = 0.000000001
a1 = (1 - alpha) / alpha
phi = 1.5
np.random.seed(0)


def test_distances():
    centroids = np.array([
        [1, 2],
        [3, 4]
    ])
    W_mah = np.linalg.inv(np.cov(d, rowvar=False))
    W_euc = np.identity(d.shape[1])
    assert np.isclose(_calc_dist(d, centroids, W_euc, 'euclidean'), dist_euc).all()
    assert np.isclose(_calc_dist(d, centroids, W_mah, 'mahalanobis'), dist_mah).all()


def test_update_centroids():
    U, Ue = _calc_membership(dist_mah, phi, a1)
    Uphi = U ** phi
    Uephi = Ue ** phi
    assert np.isclose(_update_centroids(Uphi, Uephi, a1, dist_mah, d),
                      np.array([[1.16678848, 1.00002121], [1.17421676, 0.9296254]])).all()


def test_cluster_with_distances():
    FKMEx(2, 1.5, 'euclidean', alpha=0.755067750159772).fit(d)
    FKMEx(2, 1.5, 'diagonal', alpha=0.755067750159772).fit(d)


def test_integration():
    np.random.seed(0)
    k2 = FKMEx(2, 1.5, 'mahalanobis', alpha=alpha)
    k2.fit(d)

    PIC = k2._PIC(np.array([0., 1, 1]), np.array([0., 0, 0]))
    assert np.isclose(PIC,
                      [[0.025, 0.975], [2, 2]]).all()


def test_repr():
    k2 = FKMEx(2, 1.5, 'mahalanobis', alpha=alpha)
    assert repr(k2) == '<FKMEx classes=2 phi=1.5 dist=mahalanobis>'
    k2.fit(d)
    assert repr(k2) == '<FKMEx (fitted) classes=2 phi=1.5 dist=mahalanobis>'
