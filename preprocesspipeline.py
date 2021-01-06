import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

#Categorical missing value imputer
class CategoricalImputer(BaseEsimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].fillna('Missing')

        return x
