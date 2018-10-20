import pytest
import numpy as np
import pandas as pd
import pygmo as pg

from pyPTF.ptf import PTF
from pyPTF.symb_functions import add, sub, inv, div
from pyPTF.fkmeans import FKMEx
from pyPTF.utils import summary
from pyPTF.optim import PTFOptim

latex_lines = ['\\begin{tabular}{ccccc}',
               '\\toprule',
               '{} & x & z & PI_{L} & PI_{U} \\\\',
               '\\midrule',
               'Clusters & \\multicolumn{2}{c}{Centroids} & \\multicolumn{2}{c}{Cluster residuals} \\\\',
               '\\cmidrule(rl){2-3} \\cmidrule(rl){4-5}',
               '1 & 3.02 & 1.98 & 0.00 & 0.00 \\\\',
               '2 & 4.00 & 1.00 & 0.00 & 0.00 \\\\',
               'Eg & -- & -- & 0.00 & 0.00 \\\\',
               '\\bottomrule',
               '\\end{tabular}']

x = np.array([1, 2, 3, 4])
y = x ** 2
z = np.array([4, 3.5, 2, 1])
d = pd.DataFrame({'x': x, 'y': y, 'z': z})
formula = 'y~x+z'


def test_gp_blocks():
    assert add(1, 1) == 2
    assert sub(1, 1) == 0
    assert div(1, 1) == 1
    assert inv(2) == 0.5


def test_use_all_columns():
    assert PTF(d, 'y~.', {'generations': 1}, simplify=True).xs == ['x', 'z']


def test_integration():
    test_ptf = PTF(d, formula, {'generations': 1}, simplify=True)

    assert test_ptf.to_latex() is None
    assert repr(test_ptf) == '<PTF(y~x+z): Not trained>'

    test_ptf.fit()
    assert repr(test_ptf.ptf) == 'x**2'
    assert repr(test_ptf) == '<PTF(y~x+z): x**2>'
    assert all(test_ptf.predict(d) == y)

    k2 = FKMEx(2, 1.5, 'mahalanobis', alpha=0.755067750159772)
    # k2 = FKMEx(2, 1.5, 'mahalanobis')  # Can't run this with setup.py test
    k2.fit(test_ptf)
    test_ptf.add_uncertainty(k2)
    assert test_ptf.predict(d).sum() == 90.0

    per_char = [a == b for a, b in zip([x for x in summary(test_ptf)],
                                       [x for x in '\n'.join(latex_lines)])]
    assert np.sum(per_char) / len(per_char) > 0.95

    assert test_ptf.to_latex() == 'x^{2}'
    assert str(test_ptf.simplify_program(test_ptf.gp_estimator._program)) == 'x**2'

    test_ptf.add_uncertainty(None)

    k2 = FKMEx(2, 1.5, 'mahalanobis')
    test_ptf.cleaned_data = None
    with pytest.raises(Exception):
        k2.fit(test_ptf)
