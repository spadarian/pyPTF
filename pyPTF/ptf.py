"""Module to generate a pedotransfer functions."""

import logging

import numpy as np
from scipy.stats import linregress
from gplearn.genetic import SymbolicRegressor
from gplearn._program import _Program
from sympy import (symbols, simplify, latex, Float, preorder_traversal, sympify,
                   sin, cos, tan, Abs, Max, Min, Mul)

from .symb_functions import div, sub, add, inv

logger = logging.getLogger(__name__)


class PTF(object):
    """Pedotrasfer function using Symbolic Regression (Genetic Algorithms).

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with the data to train the PTF.
    formula : str
        Formula in the format 'y~x1+x2'. The variable names should be included
        in `data`. Use 'y~.' to select all the columns (expect `y`) as predictors.
    sym_kwargs : dict
        Extra arguments for the gplearn.genetic.SymbolicRegressor (the default is {}).
    simplify : bool
        Simplify the final PTF (the default is True).

    Attributes
    ----------
    cleaned_data : pd.DataFrame
        Subset of `data` only using the selected columns (`xs` and `y`).
        The rows containing NAs are dropped.
    xs : list
        Name of the independant variables.
    y : str
        Name of the dependant variables.
    gp_estimator : gplearn.genetic.SymbolicRegressor
        Instance of SymbolicRegressor.
    stats : dict
        Training statistics (R², RMSE).
    trained : bool
        If `gp_estimator` has been trained or not.
    uncertainty : {None, gplearn._program._Program}
        Uncertainty information. It should be added using the :func:`~PTF.add_uncertainty` method.
    ptf :
        PTF as a sympy expression.
    data
    formula
    simplify

    """
    def __init__(self, data, formula, sym_kwargs={}, simplify=True):
        self.data = data
        self.cleaned_data = None
        self.formula = formula
        self.xs = []
        self.y = None
        self._parse_formula()
        self._clean_data()
        self.gp_estimator = None
        self._init_gp(sym_kwargs)
        self.stats = {}
        self.trained = False
        self.simplify = simplify
        self.uncertainty = None
        self.ptf = None

    @property
    def ptf(self):
        return self.__ptf

    @ptf.setter
    def ptf(self, program):
        if self.trained:
            assert isinstance(program, _Program), \
                'The ptf should be a gplearn program'
        ptf = self.to_symb(program)
        if self.simplify:
            ptf = simplify(ptf)
        self.__ptf = ptf

    def _parse_formula(self):
        y, xs = self.formula.split('~')
        y = y.strip()
        xs = xs.strip()
        if xs == '.':
            xs = list(self.data.columns)
            xs.remove(y)
        else:
            xs = [x.strip() for x in xs.split('+')]
        self.xs = xs
        self.y = y

    def _clean_data(self):
        self.cleaned_data = self.data[self.xs + [self.y]].dropna()

    def _init_gp(self, sym_kwargs):
        default_params = {
            'population_size': 5000,
            'generations': 10,
            'stopping_criteria': 0.01,
            'p_crossover': 0.7,
            'p_subtree_mutation': 0.1,
            'p_hoist_mutation': 0.05,
            'p_point_mutation': 0.1,
            'max_samples': 0.9,
            'verbose': 1,
            'parsimony_coefficient': 0.0001,
            'n_jobs': -1
        }

        default_params.update(sym_kwargs)
        est_gp = SymbolicRegressor(**default_params)
        self.gp_estimator = est_gp

    def fit(self):
        X = self.cleaned_data[self.xs]
        Y = self.cleaned_data[self.y]
        try:
            self.gp_estimator.fit(X, Y)
        except KeyboardInterrupt:
            last_programs = self.gp_estimator._programs[-1]
            with_fitness = [(x.raw_fitness_, x) for x in last_programs]
            with_fitness.sort(key=lambda x: x[0])
            self.ptf = with_fitness[0][1]
            self.gp_estimator._program = with_fitness[0][1]
            msg = 'Training stoped at generation {}'.format(len(self.gp_estimator._programs) - 1)
            logger.warning(msg)
        self.trained = True

        pred = self.gp_estimator._program.execute(X.values)
        slope, intercept, r_value, p_value, std_err = linregress(pred,
                                                                 Y)
        rmse = np.sqrt(np.mean((pred - Y) ** 2))
        stats = {
            'R2': r_value ** 2,
            'RMSE': rmse
        }
        self.stats = stats

        self.ptf = self.gp_estimator._program
        return self

    def predict(self, X):
        """Predict target variable.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame to predict. It should contain the columns `xs`.

        Returns
        -------
        np.ndarray
            Predicted values.

        """
        filtered = X[self.xs]
        pred = self.gp_estimator._program.execute(filtered.values)
        if self.uncertainty:
            m = self.uncertainty.membership(filtered)
            PL = np.matmul(m, self.PIC) + pred.reshape(-1, 1)
            pred = np.array([PL[:, 0], pred, PL[:, 1]]).T
        return pred

    def to_symb(self, program):
        """Convert a gplearn program to sympy expression."""
        neg = np.negative

        locals = {
            'sin': sin,
            'cos': cos,
            'tan': tan,
            'abs': Abs,
            'max': Max,
            'min': Min,
            'mul': Mul,
            'neg': neg,
            'div': div,
            'sub': sub,
            'add': add,
            'inv': inv,
        }

        expressions = {"X{}".format(i): symbols(sym)
                       for i, sym in enumerate(self.xs)
                       if 'X{}'.format(i) in str(program)}
        locals.update(expressions)
        ptf = sympify(str(program), locals=locals)
        return ptf

    def simplify_program(self, program):
        """Transform a glplearn program to sympy exmpression and simplify."""
        ptf = self.to_symb(program)
        ptf = simplify(ptf)
        return ptf

    def to_latex(self):
        """Format the ptf formula as a LaTeX equation."""
        if self.ptf:
            text = latex(self.ptf)
        else:
            text = None
        return text

    def __repr__(self):
            main_repr = '<{}({}): {{}}>'.format(self.__class__.__name__,
                                                self.formula)
            if self.ptf is None:
                repr_ = main_repr.format('Not trained')
            else:
                # Round floats
                decimals = 3
                exp2 = self.ptf
                for e in preorder_traversal(self.ptf):
                    if isinstance(e, Float):
                        exp2 = exp2.subs(e, round(e, decimals))

                symb = str(exp2)
                if len(symb) > 23:
                    symb = symb[:20] + '...'
                repr_ = main_repr.format(symb)
            return repr_

    def add_uncertainty(self, fkme, conf=0.95):
        """Add uncertainty data from a :class:`~.fkmeans.FKMEx` object.

        Parameters
        ----------
        fkme : :class:`~.fkmeans.FKMEx`
            Fitted fuzzy k-means with extragrades class.
        conf : float
            Confidence level used when estimimating the prediction_interval (the default is 0.95).

        """
        try:
            obs = self.cleaned_data[self.y].values
            if self.uncertainty is not None:
                self.uncertainty = None
            pred = self.predict(self.cleaned_data)
            self.uncertainty = fkme
            self.PIC = fkme._PIC(obs, pred, conf)
            self.MPI = fkme.MPI(obs, pred, conf)
            self.PICP = fkme.PICP(obs, pred, conf)
            logger.info('Uncertainty info successfully added to the PTF')
        except Exception:
            pass
