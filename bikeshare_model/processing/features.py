from typing import List
import sys
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder


class WeekdayImputer(BaseEstimator, TransformerMixin):
    """Impute missing values in 'weekday' column by extracting dayname from 'dteday' column"""

    def __init__(self, variable: str, date_var: str):

        if not isinstance(variable, str):
            raise ValueError("variable name should be a string")
        if not isinstance(date_var, str):
            raise ValueError("date variable name should be a string")

        self.variable = variable
        self.date_var = date_var

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # convert 'dteday' column to Datetime datatype
        X[self.date_var] = pd.to_datetime(X[self.date_var], format="%Y-%m-%d")

        wkday_null_idx = X[X[self.variable].isnull() == True].index
        X.loc[wkday_null_idx, self.variable] = (
            X.loc[wkday_null_idx, self.date_var].dt.day_name().apply(lambda x: x[:3])
        )

        # drop 'dteday' column after imputation
        X.drop(self.date_var, axis=1, inplace=True)

        return X


class WeathersitImputer(BaseEstimator, TransformerMixin):
    """Impute missing values in 'weathersit' column by replacing them with the most frequent category value"""

    def __init__(self, variable: str):

        if not isinstance(variable, str):
            raise ValueError("variable name should be a string")

        self.variable = variable

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        X = X.copy()
        self.fill_value = X[self.variable].mode()[0]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.variable] = X[self.variable].fillna(self.fill_value)

        return X


class Mapper(BaseEstimator, TransformerMixin):
    """
    Ordinal categorical variable mapper:
    Treat column as Ordinal categorical variable, and assign values accordingly
    """

    def __init__(self, variable: str, mappings: dict):

        if not isinstance(variable, str):
            raise ValueError("variable name should be a string")

        self.variable = variable
        self.mappings = mappings

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.variable] = X[self.variable].map(self.mappings).astype(int)

        return X


class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Change the outlier values:
        - to upper-bound, if the value is higher than upper-bound, or
        - to lower-bound, if the value is lower than lower-bound respectively.
    """

    def __init__(self, variable: str):

        if not isinstance(variable, str):
            raise ValueError("variable name should be a string")

        self.variable = variable

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        X = X.copy()
        q1 = X.describe()[self.variable].loc["25%"]
        q3 = X.describe()[self.variable].loc["75%"]
        iqr = q3 - q1
        self.lower_bound = q1 - (1.5 * iqr)
        self.upper_bound = q3 + (1.5 * iqr)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        for i in X.index:
            if X.loc[i, self.variable] > self.upper_bound:
                X.loc[i, self.variable] = self.upper_bound
            if X.loc[i, self.variable] < self.lower_bound:
                X.loc[i, self.variable] = self.lower_bound

        return X


class WeekdayOneHotEncoder(BaseEstimator, TransformerMixin):
    """One-hot encode weekday column"""

    def __init__(self, variable: str):

        if not isinstance(variable, str):
            raise ValueError("variable name should be a string")

        self.variable = variable
        self.encoder = OneHotEncoder(sparse_output=False)

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        X = X.copy()
        self.encoder.fit(X[[self.variable]])
        # Get encoded feature names
        self.encoded_features_names = self.encoder.get_feature_names_out(
            [self.variable]
        )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        encoded_weekdays = self.encoder.transform(X[[self.variable]])
        # Append encoded weekday features to X
        X[self.encoded_features_names] = encoded_weekdays

        # drop 'weekday' column after encoding
        X.drop(self.variable, axis=1, inplace=True)

        return X
