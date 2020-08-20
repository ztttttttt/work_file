from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, Imputer
import numpy as np
import pandas as pd


class FeaturesEncoding(BaseEstimator, TransformerMixin):

    def __init__(self, max_num=20, num_cols_strategy='mean', cat_cols_strategy='ordinal', label_col='label_class'):
        self.num_cols_strategy = num_cols_strategy
        self.cat_cols_strategy = cat_cols_strategy
        self.max_num = max_num
        self.label_col = label_col

        self.numeric_columns = ['int', 'float', 'float64', 'int64']

        self.cat_cols = None
        self.num_cols = None
        self.useful_cols = None
        self._num_imputer = None
        self._label_encoders = None
        self.categories_ = None
        self.one_hot = None

    def fit(self, X, y=None):

        # abandon illegal cols
        X = X.replace('', np.nan)
        X = X.replace('N/A', np.nan)
        # try to convert to numerical type
        X = X.apply(pd.to_numeric, errors='ignore')

        abandoned_cols = []
        for col in X.columns:
            if len(set(X[col].map(lambda x: type(x))) & set({list, dict, tuple})) != 0:
                abandoned_cols.append(col)
            elif X[col].unique().shape[0] <= 1:
                abandoned_cols.append(col)
            elif X[col].dtype not in self.numeric_columns and X[col].unique().shape[0] > self.max_num:
                abandoned_cols.append(col)

        X_ = X.drop(abandoned_cols, axis=1)

        self.cat_cols = X_.select_dtypes(exclude=self.numeric_columns).columns
        self.num_cols = X_.select_dtypes(include=self.numeric_columns).columns
        self.useful_cols = self.num_cols.append(self.cat_cols)

        # impute num type cols, strategy -2 means identification
        if self.num_cols_strategy == -2:
            pass
        else:
            self._num_imputer = Imputer(strategy=self.num_cols_strategy)
            X_num = X[self.num_cols].values
            self._num_imputer.fit(X_num)

        # impute and encode cat type cols
        X_ = X_[self.cat_cols]
        n_samples, n_features = X_.shape

        self._label_encoders = [LabelEncoder() for i in range(n_features)]

        for i in range(n_features):
            X_i = X_.iloc[:, i].dropna()
            label_encoder = self._label_encoders[i]
            label_encoder.fit(X_i)

        self.categories_ = [le.classes_ for le in self._label_encoders]

        # ordinal strategy encode
        if self.cat_cols_strategy == 'ordinal':
            return self

        # one-hot strategy
        X_int = np.ones_like(X_, dtype=np.int)

        for i in range(n_features):
            Xi = X_.iloc[:, i].fillna('NaN')
            valid_mask = np.in1d(Xi, self.categories_[i])
            X_int[~valid_mask, i] = 999
            X_int[valid_mask, i] = self._label_encoders[i].transform(Xi[valid_mask])

        self.one_hot = OneHotEncoder(handle_unknown='ignore', sparse=False)
        self.one_hot.fit(X_int)

        return self

    def transform(self, X, y=None):

        # delete null label rows
        if y is not None:
            df = pd.concat([X, y], axis=1)
            df = df.dropna(axis=self.label_col, how='any')
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]

        lacked_cols = set(self.useful_cols) - set(X.columns)
        if lacked_cols is not None:
            for col in lacked_cols:
                X.loc[:, col] = np.nan

        # impute num cols
        num = X.loc[:, self.num_cols].copy()
        num = num.apply(pd.to_numeric, errors='coerce')

        if self.num_cols_strategy == -2:
            X_num_transformed = num.fillna(-2)

        else:
            X_num_transformed = self._num_imputer.transform(num)

        # impute and encode cat type cols
        X_ = X[self.cat_cols]
        n_samples, n_features = X_.shape

        X_int = np.ones_like(X_, dtype=np.int)

        for i in range(n_features):
            Xi = X_.iloc[:, i].fillna('NaN')
            valid_mask = np.in1d(Xi, self.categories_[i])
            X_int[~valid_mask, i] = -1
            X_int[valid_mask, i] = self._label_encoders[i].transform(Xi[valid_mask])

        # concatenate features
        if self.cat_cols_strategy == 'ordinal':
            X_transformed = np.concatenate((X_num_transformed, X_int), axis=1)
            return X_transformed

        X_int[X_int == -1] = 999
        X_one_hot = self.one_hot.transform(X_int)
        X_onehot_transformed = np.concatenate((X_num_transformed, X_one_hot), axis=1)

        return X_onehot_transformed
