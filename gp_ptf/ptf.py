import logging

import numpy as np
from scipy.stats import linregress
from gplearn.genetic import SymbolicRegressor
from sympy import symbols, simplify, latex, Float, preorder_traversal

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
        Training statistics (R2, RMSE).
    trained : bool
        If `gp_estimator` has been trained or not.
    symb : sympy.core.mul.Mul
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
        self.parse_formula()
        self.clean_data()
        self.gp_estimator = None
        self.init_gp(sym_kwargs)
        self.stats = {}
        self.trained = False
        self.simplify = simplify
        self.symb = None
        self.uncertainty = None

    def parse_formula(self):
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

    def clean_data(self):
        self.cleaned_data = self.data[self.xs + [self.y]].dropna()

    def init_gp(self, sym_kwargs):
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
        self.gp_estimator.fit(X, Y)
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

        self.init_symb()
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
        from gp_ptf.symb_functions import div, sub, add, inv
        # from sympy import add, mul, sqrt
        # add = add.Add
        # mul = mul.Mul
        from sympy import sin, cos, tan
        from sympy import Abs as abs
        from sympy import Max as max
        from sympy import Min as min
        from sympy import Mul as mul
        neg = np.negative

        expressions = ["X{} = symbols('{}')".format(i, sym)
                       for i, sym in enumerate(self.xs)
                       if 'X{}'.format(i) in str(program)]
        for ex in expressions:
            exec(ex)

        ptf = eval(str(program))
        return ptf

    def simplify_program(self, program):
        ptf = self.to_symb(program)
        ptf = simplify(ptf)
        return ptf

    def init_symb(self):
        ptf = self.to_symb(self.gp_estimator._program)
        if self.simplify:
            ptf = simplify(ptf)
        self.symb = ptf

    def to_latex(self):
        if self.symb:
            text = latex(self.symb)
        else:
            text = ''
        return text

    def __repr__(self):
            main_repr = '<{}({}): {{}}>'.format(self.__class__.__name__,
                                                self.formula)
            if self.symb is None:
                repr_ = main_repr.format('Not trained')
            else:
                # Round floats
                decimals = 3
                exp2 = self.symb
                for e in preorder_traversal(self.symb):
                    if isinstance(e, Float):
                        exp2 = exp2.subs(e, round(e, decimals))

                symb = str(exp2)
                if len(symb) > 23:
                    symb = symb[:20] + '...'
                repr_ = main_repr.format(symb)
            return repr_

    def add_uncertainty(self, fkme, conf=0.95):
        try:
            obs = self.cleaned_data[self.y].values
            pred = self.predict(self.cleaned_data)
            # uncertainty = {
            #     'PIC': fkme.PIC(obs, pred),
            #     'centroids': fkme.centroids,
            #     'W': fkme.W,
            #     'disttype': fkme.disttype,
            #     'phi': fkme.phi,
            #     'alpha': fkme.alpha,
            # }
            self.uncertainty = fkme
            self.PIC = fkme.PIC(obs, pred, conf)
            self.MPI = fkme.MPI(obs, pred, conf)
            self.PICP = fkme.PICP(obs, pred, conf)
            logger.info('Uncertainty info successfully added to the PTF')
        except Exception:
            pass
