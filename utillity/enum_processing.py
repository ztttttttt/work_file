from sklearn.base import BaseEstimator,TransformerMixin
import pandas as pd
import numpy as np
from functools import partial
import logging


class EnumMapper(BaseEstimator, TransformerMixin):
    '''
    select effective columns and transform enum column
    '''

    def __init__(self, maximum_enum_num):
        '''
        maximum_enum_num: maximum number of unique feature value or will remove this feature
        '''
        self.maximum_enum_num = maximum_enum_num

        self.enum_col_map_dk = None  # mapping column dict
        self.clean_col_names = None  # all used columns
        self.cleaned_col_types = None  # all used columns types

        self.__useful_column_agg = None

        self.true_value_set = {u'是', u'Yes', u'yes', True, u'True', u'TRUE'}
        self.false_value_set = {u'否', u'No', u'no', False, u'False', u'FALSE'}
        self.true_default_val = 1
        self.false_default_val = -1

    def fit(self, df,y=None):
        '''compute to get mapping column dict and selected columns'''
        self.enum_col_map_dk = {}
        self.__useful_column_agg = []
        self.clean_col_names = None

        not_number_cols = df.select_dtypes(exclude=['int', 'float']).columns.tolist()
        number_cols = df.select_dtypes(include=['int', 'float']).columns.tolist()


        for col in not_number_cols:
            try:
                uniq_val_arr = pd.unique(df[col])
            except TypeError as e:
                continue

            uniq_length = uniq_val_arr.shape[0]

            if uniq_length <= self.maximum_enum_num:  # aggregate selected col
                self.__useful_column_agg.append(col)

                col_val_mapped_dk = self.__make_col_map_val(uniq_val_arr)
                self.enum_col_map_dk[col] = col_val_mapped_dk
            else:
                pass

        # add columns name of number col to useful 'not_number' cols
        self.clean_col_names = sorted(self.__useful_column_agg + number_cols)

        # record the date types of each columns
        cleaned_col_types = []
        for col in self.clean_col_names:
            if df[col].dtype == np.float:
                cleaned_col_types.append('float')
            elif df[col].dtype == np.int:
                cleaned_col_types.append('int')
            elif df[col].dtype == np.object:
                cleaned_col_types.append('object')
            elif df[col].dtype == np.bool:
                cleaned_col_types.append('bool')
            else:
                cleaned_col_types.append('unknown')
        self.cleaned_col_types = cleaned_col_types
        return self

    def transform(self, df):
        if self.enum_col_map_dk is None or self.clean_col_names is None or self.cleaned_col_types is None:
            raise Exception('please fit first!!')

        # create a empty to store df value
        out_dict = {}
        for col, col_types in zip(self.clean_col_names, self.cleaned_col_types):
            # check whether col in df.columns
            if col not in df.columns:
                out_dict[col] = np.full((df.shape[0]), np.nan)
            else:
                if col_types == 'float' or col_types == 'int':
                    # convert col to number type, 'coerce' will set the col to nan if cannot convert
                    out_dict[col] = pd.to_numeric(df[col], errors='coerce')

                elif col_types == 'object' or col_types == 'bool':
                    # map enum value use number
                    map_val_dk = self.enum_col_map_dk[col]
                    partial_f = partial(self.__do_map_enum, map_val_dk)

                    out_dict[col] = df[col].map(partial_f)

                else:  # unknown data type,set a default value
                    logging.warning('get unknown data type for column:{} data type:{}'.format(col, col_types))

                    out_dict[col] = np.full((df.shape[0]), 0)

        df_all_col = pd.DataFrame(out_dict, columns=self.clean_col_names)
        return df_all_col

    def __do_map_enum(self, map_val_dk, x):
        if pd.isnull(x):  # the value is nan, just return nan
            return np.nan

        if x in map_val_dk.keys():  # the value in mapping json
            return map_val_dk[x]
        else:  # the value not in mapping json
            return -2

    def __make_col_map_val(self, uniq_val_arr):
        original_val = []  # to store original value
        col_map_val = []  # to store mapped value

        start_counter = 2  # start mapping value except for true/false set
        for val in uniq_val_arr:
            if pd.isnull(val):
                #                 col_map_val.append(np.nan)
                pass
            else:
                original_val.append(val)
                if (val in self.true_value_set) and type(val) != np.dtype(int) and type(val) != np.dtype(float):
                    col_map_val.append(self.true_default_val)  # default mapping value for true set
                elif (val in self.false_value_set) and type(val) != np.dtype(int) and type(val) != np.dtype(float):
                    col_map_val.append(self.false_default_val)  # default mapping value for false set
                else:
                    col_map_val.append(start_counter)
                    start_counter += 1

        return dict(zip(original_val, col_map_val))
