import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeClassifier


class WoeMapper(BaseEstimator, TransformerMixin):
    def __init__(self, only_trans_str=True, max_str_values=1000, max_binning_tree_depth=4, min_bin_pct=0.02,
                 min_missing_pct_calc_woe=0.02, verbose=0):
        '''
        only_trans_str - if False, all columns include numeric ones would be binned and transformed.
        max_str_values - if a columns is str type and have over max_str_values unique values, it would be discarded.
        max_bin_depth - Controls the max_depth of the decision tree used for binning.
        pct_min_bin - A bin whose percentage is under min_bin_pct should be merged.
        min_missing_pct_calc_woe - when the percentage of missing value in a columns(numeric) is less than it, the missings will not be treated as a singe bin.
                                   Instead, it will be filled with mean value.
        verbose - If positive, print logs during fitting and transformation.
        '''
        self.only_trans_str = only_trans_str
        self.max_str_values = max_str_values
        self.max_binning_tree_depth = max_binning_tree_depth
        self.min_bin_pct = min_bin_pct
        self.min_missing_pct_calc_woe = min_missing_pct_calc_woe
        self.verbose = verbose

        self.is_fit = False

        # A dict stored all info of each column
        self.dct_col_result = {}
        self.vars_str_to_float = [
            'X_91ZX_loanInfo_size',
            'X_IdCardAge',
            'X_JD_Latitude',
            'X_JD_Longtitude',
            'X_JD_characteristic',
            'X_JD_identity',
            'X_JD_performance',
            'X_JD_property',
            'X_JD_relationship_network',
            'X_MX_raw_data_packages_items_total',
            'X_MX_raw_data_packages_items_used',
            'X_MX_raw_report_application_check_check_points_age',
            'X_MX_raw_report_cell_behavior_behavior_call_in_time',
            'X_MX_raw_report_cell_behavior_behavior_call_out_cnt',
            'X_Mobile_DurationOfOnline',
            'X_SHZX_ApplyMoney',
            'X_SHZX_ApplyMonth',
            'X_XINYAN_Black_results_detail_max_overdue_amt',
            'X_XINYAN_Black_results_detail_max_overdue_days',
            'x_ALISCORE',
            'X_DNA_Behavior_last_settled_to_apply_days',
            'X_DNA_Behavior_historical_overdue_7_days_cnt',
            'X_DNA_Behavior_historical_overdue_15_days_cnt',
            'X_DNA_Behavior_historical_overdue_30_days_cnt',
            'X_DNA_Behavior_historical_overdue_7_days_ratio',
            'X_DNA_Behavior_historical_overdue_15_days_ratio',
            'X_DNA_Behavior_historical_overdue_30_days_ratio',
            'X_DNA_Behavior_last_overdue_avg_days',
            'X_DNA_Behavior_historical_overdue_avg_days',
            'X_DNA_Behavior_insurance_this_purchase_apply',
            'X_DNA_Behavior_insurance_last_purchase_apply',
            'X_DNA_Behavior_insurance_first_purchase_apply',
            'X_DNA_Behavior_insurance_historical_purchase_ratio',
            'X_DNA_Behavior_insurance_last_purchase_success',
            'X_DNA_Behavior_insurance_first_purchase_success',
            'X_DNA_Behavior_insurance_historical_success_ratio',
            'X_DNA_Behavior_repayment_last_avg_days',
            'X_DNA_Behavior_repayment_last_accumulate_days',
            'X_DNA_Behavior_repayment_last_firstinstalment_days',
            'X_DNA_Behavior_repayment_last_lastinstalment_days',
            'X_DNA_Behavior_repayment_last_max_days',
            'X_DNA_Behavior_repayment_last_min_days',
            'X_DNA_Behavior_repayment_first_avg_days',
            'X_DNA_Behavior_repayment_first_accumulate_days',
            'X_DNA_Behavior_repayment_first_firstinstalment_days',
            'X_DNA_Behavior_repayment_first_lastinstalment_days',
            'X_DNA_Behavior_repayment_first_max_days',
            'X_DNA_Behavior_repayment_first_min_days',
            'X_DNA_Behavior_repayment_historical_avg_days',
            'X_DNA_Behavior_repayment_historical_accumulate_days',
            'X_DNA_Behavior_repayment_historical_firstinstalment_days',
            'X_DNA_Behavior_repayment_historical_lastlinstalment_days',
            'X_DNA_Behavior_repayment_historical_max_days',
            'X_DNA_Behavior_repayment_historical_min_days',
            'X_DNA_Behavior_application_lastinstalment_cnts',
            'X_DNA_Behavior_application_firstinstalment_cnts',
            'X_DNA_Behavior_application_historical_success_ratio',
            'X_DNA_Behavior_application_historical_faild_ratio',
            'X_DNA_Behavior_loan_lastapplication_success',
            'X_DNA_Behavior_loan_historical_success_ratio',
            'X_DNA_Behavior_hesitation_firstinstallment',
            'X_DNA_Behavior_hesitation_lastinstallment',
            'X_DNA_Behavior_hesitation_historical_ratio',
            'X_DNA_Model_first_opscore',
            'X_DNA_Model_first_p_value',
            'X_DNA_Model_last_opscore',
            'X_DNA_Model_last_p_value',
            'X_DNA_Behavior_diffamount_first',
            'X_DNA_Behavior_last_withdraw_ratio',
            'X_DNA_Behavior_first_withdraw_ratio',
            'X_DNA_Behavior_historical_withdraw_ratio',
            'X_DNA_Behavior_last_withdraw_ratio_diff',
            'X_DNA_Behavior_first_withdraw_ratio_diff',
            'X_DNA_Behavior_historical_withdraw_ratio_diff',
            'X_DNA_Behavior_this_pmt',
            'X_DNA_Behavior_last_pmt',
            'X_DNA_Behavior_first_pmt',
            'X_DNA_Behavior_historical_pmt',
            'X_DNA_Behavior_last_pmt_diff',
            'X_DNA_Behavior_first_pmt_diff',
            'X_DNA_Behavior_historical_pmt_diff',
            'X_DNA_Behavior_historical_overdue_cnts',
            'X_DNA_Behavior_last_firstinstalment_overdue_days',
            'X_DNA_Behavior_first_firstinstalment_overdue_days',
            'X_DNA_Behavior_register2application_diffdays',
            'X_DNA_Behavior_credit2firstloan_diffdays',
            'X_DNA_Behavior_lastloan2thisloan_diffdays',
            'X_DNA_Behavior_historical_max_diffdays',
            'X_DNA_Behavior_historical_min_diffdays',

            'X_JD_CreditLine',
            'X_JD_Principal',
            'X_JD_WithdrawRatio',
            'X_prev_max_overdue_day',
            'X_prev_one_overdue_day',
            'X_history_max_overdue_day',
            'X_JD_DiffAmount',
            'X_JD_DiffRatio',
            'X_JD_DiffLoanDay',
            'X_JD_DiffCreditRatio',
            'X_JD_AdvanceRatio',
            'X_JD_PrevAppAddvance',
            'X_JD_PrevOverdueDays',
            'X_JD_IntervalDays',
            'X_JD_PrevRepayTimes',
            'X_JD_AvgPrincipal',
            'X_JD_PrevOverdueTimes',
            'X_JD_PrevUnOverdueTimes',
            'X_JD_PrevOverdue7DayTimes',
            'X_JD_MaxOverdueTimes',
            'X_JD_LoanTimes',
            'X_DNA_Behavior_submit_hour',
            'X_Graph_Closeness',
            'X_MX_raw_data_families_family_num',
            'X_JD_histRepayTimes']

    # super().__init__()

    def fit(self, df, y):
        df = df.copy()
        for col in self.vars_str_to_float:
            if col in df.columns:
                df.loc[:, col] = pd.to_numeric(df[col], errors='coerce').astype(float)
        # 1.Get column types
        self.get_column_type(df)
        ls_numeric_cols = []
        ls_not_numeric_cols = []
        for col in self.dct_col_result.keys():
            if self.dct_col_result[col]['dtype'] in ('int', 'float'):
                ls_numeric_cols.append(col)
            else:
                ls_not_numeric_cols.append(col)

        # 2.Among not numeric columns, discard columns whose value_count is over max_str_values.

        for col in ls_not_numeric_cols:
            df[col].fillna('NA', inplace=True)
            self.dct_col_result[col]['ls_origin_values_NA'] = list(set(df[col]))
            if df[col].value_counts().count() > self.max_str_values:
                self.dct_col_result[col]['valid'] = False

        # 3.Bins whose percentage under min_bin_pct should be merged as one bin.
        # 4.Map the bin to the good rate.
        df_preprocesed = self.preprocess_not_numeric_columns(df, y)

        # 5.Use decision tree to do binning
        if self.only_trans_str:
            ls_cols_to_bin = [col for col in self.dct_col_result.keys() if
                              self.dct_col_result[col]['valid'] and not self.dct_col_result[col]['dtype'] in (
                              'int', 'float')]
        else:
            ls_cols_to_bin = [col for col in self.dct_col_result.keys() if self.dct_col_result[col]['valid']]
        self.auto_binning(df_preprocesed, y, ls_cols_to_bin, self.max_binning_tree_depth)
        return self

    def transform(self, df):
        df = df.copy()
        for col in self.vars_str_to_float:
            if col in df.columns:
                df.loc[:, col] = pd.to_numeric(df[col], errors='coerce').astype(float)
        if not self.is_fit:
            raise Exception('Please fit before transform!')
        i = 1
        for col in self.dct_col_result.keys():
            if self.dct_col_result[col]['valid']:
                if not col in df.columns:
                    df.loc[:, col] = np.nan
                i += 1
                # Transform non-numeric columns
                if not self.dct_col_result[col]['dtype'] in ('int', 'float'):
                    df[col].fillna('NA', inplace=True)
                    df.loc[:, col] = df[col].astype(object).replace(self.dct_col_result[col]['cat_woe_mapper'])
                # Transform numeric columns
                elif not self.only_trans_str:
                    if self.dct_col_result[col]['mean_fillna']:
                        df[col].fillna(self.dct_col_result[col]['mean_fillna'], inplace=True)
                        sr_bin = pd.cut(df[col], self.dct_col_result[col]['bin_threshold'], right=False)
                        df.loc[:, col] = sr_bin.replace(self.dct_col_result[col]['bin_woe_mapper'])
                    else:
                        self.dct_col_result[col]['sr_0'] = df[col]
                        sr_bin = pd.cut(df[col], self.dct_col_result[col]['bin_threshold'], right=False)
                        sr_bin = sr_bin.cat.add_categories(['NA'])
                        sr_bin.fillna('NA', inplace=True)
                        df.loc[:, col] = sr_bin.replace(self.dct_col_result[col]['bin_woe_mapper'])
                else:
                    continue

        self.valid_columns = self.valid_cat_columns + self.valid_num_columns
        self.clean_col_names = self.valid_columns
        for col in self.valid_columns:
            df.loc[:, col] = pd.to_numeric(df[col], errors='coerce')

        return df[self.valid_columns].fillna(0)

    def get_column_type(self, df):

        for col in df.columns:
            self.dct_col_result[col] = {}
            self.dct_col_result[col]['valid'] = True
            self.dct_col_result[col]['dtype'] = df[col].dtype

    def preprocess_not_numeric_columns(self, df, y):

        i = 0
        for col in self.dct_col_result.keys():
            i += 1
            if self.dct_col_result[col]['valid'] and not self.dct_col_result[col]['dtype'] in ('int', 'float'):
                # 1.Bins whose percentage under min_bin_pct should be merged as one bin.
                sr_col_pct = df[col].value_counts() / df.shape[0]
                sr_col_pct_min_bin_pct = sr_col_pct[sr_col_pct < self.min_bin_pct]
                if not sr_col_pct_min_bin_pct.empty:
                    self.dct_col_result[col]['mapper_min_bin_pct'] = dict(
                        zip(sr_col_pct_min_bin_pct.index, ['BIN_min_bin_pct', ] * len(sr_col_pct_min_bin_pct)))
                    df[col].replace(self.dct_col_result[col]['mapper_min_bin_pct'], inplace=True)
                else:
                    self.dct_col_result[col]['mapper_min_bin_pct'] = None

                # 2.Map the bin to the bad rate = 1-y.
                self.dct_col_result[col]['mapper_bin_to_bad'] = dict((1 - y).groupby(df[col]).mean())
                df[col].replace(self.dct_col_result[col]['mapper_bin_to_bad'], inplace=True)

        return df

    def auto_binning(self, df_preprocesed, y, ls_cols_to_bin, max_binning_tree_depth=4):
        # Columns in ls_cols_to_bin should all be numeric, including those transformed from not numeric ones.
        df_preprocesed['y'] = y
        for col in ls_cols_to_bin:
            df_col_y = df_preprocesed[[col, 'y']]
            df_col_y_notnull = df_col_y[df_col_y[col].notnull()].copy()
            if df_col_y_notnull.shape[0] == 0:
                self.dct_col_result[col]['valid'] = False
                continue
            tree_clf = DecisionTreeClassifier(criterion='entropy', max_depth=max_binning_tree_depth)
            tree_clf.fit(df_col_y_notnull[col].values.reshape(-1, 1), df_col_y_notnull['y'].values.reshape(-1, 1))
            self.dct_col_result[col]['bin_threshold'] = [-np.inf, ] + sorted(
                list(set(tree_clf.tree_.threshold[tree_clf.tree_.threshold != -2]))) + [np.inf, ]
            # 假如缺失比例小于min_missing_pct_calc_woe， 则用均值填充缺失，而不单独作为一组
            if df_col_y[col].isnull().sum() / df_col_y.shape[0] < self.min_missing_pct_calc_woe:
                self.dct_col_result[col]['mean_fillna'] = df_col_y[col].mean()
                df_col_y.loc[:, col] = df_col_y[col].fillna(self.dct_col_result[col]['mean_fillna'])
                df_col_y.loc[:, 'bin'] = pd.cut(df_col_y[col], self.dct_col_result[col]['bin_threshold'], right=False)
            else:
                df_col_y.loc[:, 'bin'] = pd.cut(df_col_y[col], self.dct_col_result[col]['bin_threshold'], right=False)
                df_col_y.loc[:, 'bin'] = df_col_y['bin'].cat.add_categories(['NA'])
                df_col_y.loc[:, 'bin'] = df_col_y['bin'].fillna('NA')
                self.dct_col_result[col]['mean_fillna'] = None
            sr_woe_mapper = self.calc_woe(df_col_y['bin'], df_col_y['y'])
            dct_bin_woe_mapper = dict(zip(sr_woe_mapper.index, sr_woe_mapper))
            self.dct_col_result[col]['bin_woe_mapper'] = dct_bin_woe_mapper

        # 数值型、非数值型的映射方不同
        # 1.数值型，数值 --> bin --> woe
        # 2.非数值型, 值 --> woe

        # 获取非数值型，值 --> woe的映射
        self.valid_cat_columns = []
        self.valid_num_columns = []
        for col in self.dct_col_result.keys():
            if self.dct_col_result[col]['valid'] and not self.dct_col_result[col]['dtype'] in ('int', 'float'):
                self.valid_cat_columns.append(col)
                df_cat_mapper = DataFrame({'origin': self.dct_col_result[col]['ls_origin_values_NA'], })
                if self.dct_col_result[col]['mapper_min_bin_pct']:
                    df_cat_mapper.loc[:, 'mapper_min_bin'] = df_cat_mapper['origin'].replace(
                        self.dct_col_result[col]['mapper_min_bin_pct'])
                    df_cat_mapper.loc[:, 'mapper_bin_bad'] = df_cat_mapper['mapper_min_bin'].replace(
                        self.dct_col_result[col]['mapper_bin_to_bad'])
                    df_cat_mapper.loc[:, 'mapper_bin_interval'] = pd.cut(df_cat_mapper['mapper_bin_bad'],
                                                                         self.dct_col_result[col]['bin_threshold'],
                                                                         right=False)
                    df_cat_mapper.loc[:, 'mapper_woe'] = df_cat_mapper['mapper_bin_interval'].replace(
                        self.dct_col_result[col]['bin_woe_mapper'])
                    self.dct_col_result[col]['cat_woe_mapper'] = dict(
                        zip(df_cat_mapper['origin'], df_cat_mapper['mapper_woe']))
                else:

                    df_cat_mapper.loc[:, 'mapper_bin_bad'] = df_cat_mapper['origin'].replace(
                        self.dct_col_result[col]['mapper_bin_to_bad'])
                    df_cat_mapper.loc[:, 'mapper_bin_interval'] = pd.cut(df_cat_mapper['mapper_bin_bad'],
                                                                         self.dct_col_result[col]['bin_threshold'],
                                                                         right=False)
                    df_cat_mapper.loc[:, 'mapper_woe'] = df_cat_mapper['mapper_bin_interval'].replace(
                        self.dct_col_result[col]['bin_woe_mapper'])
                    self.dct_col_result[col]['cat_woe_mapper'] = dict(
                        zip(df_cat_mapper['origin'], df_cat_mapper['mapper_woe']))
            elif self.dct_col_result[col]['valid'] and self.dct_col_result[col]['dtype'] in ('int', 'float'):
                self.valid_num_columns.append(col)
        self.is_fit = True

    def calc_woe(self, x_bin, y):
        df_xy = DataFrame({'x': x_bin, 'y': y})
        bad_num = sum(y)
        good_num = len(y) - bad_num
        bad_dist = df_xy.groupby('x')['y'].apply(sum) / bad_num + 0.01
        good_dist = (df_xy.groupby('x')['y'].apply(len) - df_xy.groupby('x')['y'].apply(sum)) / good_num + 0.01
        sr_woe = (bad_dist / good_dist).apply(np.log)
        # iv = round(sum((bad_dist - good_dist)*sr_woe),3)
        return sr_woe
