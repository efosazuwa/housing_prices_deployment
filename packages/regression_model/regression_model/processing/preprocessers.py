import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

#Categorical missing value imputer
class CategoricalImputer(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None) -> 'CategoricalImputer':
        return self

    def transform(self, X) -> pd.DataFrame:
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].fillna('Missing')

        return X

#Numerical missing value imputer
class NumericalImputer(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        #persist mode in a dictionary
        self.imputer_dict_ = dict()
        for feature in self.variables:
            self.imputer_dict_[feature] = X[feature].mode()[0]
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature].fillna(self.imputer_dict_[feature], inplace=True)
        return X

#Temporal variable calculator
class TemporalVariableEstimator(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None, reference_variable=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

        self.reference_variables = reference_variable

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[self.reference_variables] - X[feature]

        return X

#Frequent label categorical encoder
class RareLabelCategoricalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, tol=0.05, variables=None):
        self.tol = tol
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):

        #persist frequent labels in a dictionary
        self.encoder_dict_ = dict()

        for var in self.variables:
            #The encoder learns the most frequent categories
            t = pd.Series(X[var].value_counts())/len(X)
            #frequent labels
            self.encoder_dict_[var] = list(t[t >= self.tol].index)
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = np.where(X[feature].isin(self.encoder_dict_[feature]), X[feature], 'Rare')
        return X

#string to numbers categorical encoder
class CategoricalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y):
        temp = pd.concat([X, y], axis=1)
        temp.columns =  list(X.columns) + ['target']

        self.encoder_dict_ = {}

        for var in self.variables:
            t = temp.groupby(var)['target'].mean().sort_values(ascending=True).index
            self.encoder_dict_[var] = {k: i  for i, k in enumerate(t,0)}
        return self

    def transform(self, X):
        #encode labels
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].map(self.encoder_dict_[feature])

        #Check if transformer introduces any NaN
        if X[self.variables].isnull().any().any():
            null_counts = X[self.variables].isnull().any()
            vars_ = {key: value for key, value in null_counts.items() if value
                     is True}
            raise errors.InvalidModelInputError("Categorical encoder introduced Nan when transforming categorical variables: {}".format(vars_.keys()))
        return X

#logarithmic Transformer
class LogTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        #to accomadate pipelline
        return self

    def transform(self, X):
        X = X.copy()

        #Check that values are nonnegative for log transformation
        if not (X[self.variables]>0).all().all():
            vars_ = self.variables[(X[self.variables] <= 0).any()]
            raise errors.InvalidModelInputError("Variables contain 0 or negative values, can't apply log for vars: {}".format(vars_))

        for feature in self.variables:
            X[feature] = np.log(X[feature])

        return X

class DropUnecessaryFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, variables_to_drop=None):
        self.variables = variables_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()
        X = X.drop(self.variables, axis=1)

        return X
