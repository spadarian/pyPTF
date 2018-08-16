import numpy as np
from scipy.stats import linregress
from gplearn.genetic import SymbolicRegressor


class PTF(object):
    def __init__(self, data, formula, sym_kwargs={}):
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
