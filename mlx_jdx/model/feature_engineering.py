from sklearn.preprocessing import MinMaxScaler, Binarizer
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, Imputer
from collections import Counter
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
import json


class AddressbookFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, apply_data, word_length=[1, 2, 3], num_names_kept=100, kill_words='nb',
                 killer_num_names_kept=100, words_dict=[], max_names_num=1000):
        self.apply_data = apply_data
        # the data to calculate addressbook dict
        self.word_length = word_length
        # length of names to keep
        self.kill_words = kill_words
        # method to reduce dimension
        self.words_num_names_kept = killer_num_names_kept
        # num of words after reduce dimension
        self.num_names_kept = num_names_kept
        # num of most frequent names to keep when generating address book
        self.words_dict = words_dict
        # pre-defined names dict
        self.max_names_num = max_names_num

    def fit(self, X, y=None):

        if len(self.words_dict) > 0:
            return self

        self.names_strings = self.__dict_gen(self.apply_data)
        apply_addressbook = self.addressbook_gen(self.apply_data)
        if len(self.word_length) == 0:
            return self
        elif len(self.word_length) == 1:
            X_, dict_col_names = self.add_address_book(self.names_strings, apply_addressbook, self.word_length[0])
        else:
            dict_col_names = {}
            X_ = self.add_address_book(self.names_strings, apply_addressbook, self.word_length[0])[0]
            for i in self.word_length[1:]:
                X_add, word_dict = self.add_address_book(self.names_strings, apply_addressbook, i)
                X_ = pd.concat([X_, X_add], axis=1, join='inner')
                dict_col_names.update(word_dict)

        X_transformed = self.words_killer(train=self.apply_data, data=X_, method=self.kill_words,
                                          words_num_names_kept=self.words_num_names_kept)
        for name in X_transformed.columns:
            self.words_dict.append(dict_col_names[name])

        return self

    def transform(self, X, y=None):

        self.names_map = self.addressbook_gen(X)

        X_transformed = self.__words_col_gen(words_dict=self.words_dict, names_map=self.names_map, n='gen')[0]
        X_transformed['X_Address_only_num_ratio'] = self.names_map.map(lambda x: self.__num_addressbook_count(x))
        X_transformed['X_Address_only_letter_ratio'] = self.names_map.map(lambda x: self.__letter_addressbook_count(x))

        self.num_cols_generated = X_transformed.shape[1]

        return X.join(X_transformed)

        # return pd.concat([X, X_transformed], axis=1)
        # return np.concatenate((X, X_transformed), axis=1)

    # generate features by names appear in address book
    def add_address_book(self, names_strings, names_map, n):

        if n == 1:
            letter_c = Counter(names_strings)
            letter_dict = list(dict(letter_c.most_common()[:self.num_names_kept]).keys())
            words, dict_col_names = self.__words_col_gen(letter_dict, names_map, n)


        elif n > 1:
            words = []
            for i, letter in enumerate(names_strings[:-(n)]):
                words.append(''.join([names_strings[i: i + n]]))

            words_c = Counter(words)

            word_dict = list(dict(words_c.most_common()[:self.num_names_kept]).keys())
            words, dict_col_names = self.__words_col_gen(word_dict, names_map, n)

        return words, dict_col_names

    def words_killer(self, train, data, method, words_num_names_kept=50):
        if method == 'nb':
            normalizer = Binarizer()
            normalized_data = pd.DataFrame(normalizer.fit_transform(data))
            normalized_data.index = data.index
            train_data = pd.concat([normalized_data, train['label_class']], axis=1, join='inner')
            clf = BernoulliNB()
            clf.fit(train_data.drop('label_class', axis=1), train_data['label_class'])
            print('words killer auc: ',
                  cross_val_score(clf, train_data.drop('label_class', axis=1), train_data['label_class'],
                                  scoring='roc_auc'))
            fe = pd.Series(clf.coef_[0])
            fe.index = data.columns
            fe = fe.abs().sort_values(ascending=False)[:words_num_names_kept]
            return data[fe.index]

        if method == 'pca':
            clf = PCA(n_components=words_num_names_kept)
            train_data = pd.DataFrame(clf.fit_transform(data))
            train_data.index = data.index
            return train_data

        if method == 'lg':
            normalizer = MinMaxScaler()
            normalized_data = pd.DataFrame(normalizer.fit_transform(data))
            normalized_data.index = data.index
            train_data = pd.concat([normalized_data, train['label_class']], axis=1, join='inner')
            clf = LogisticRegression(class_weight='balanced')
            clf.fit(train_data.drop('label_class', axis=1), train_data['label_class'])
            print('words killer auc: ',
                  cross_val_score(clf, train_data.drop('label_class', axis=1), train_data['label_class'],
                                  scoring='roc_auc'))
            fe = pd.Series(clf.coef_[0])
            fe.index = data.columns
            fe = fe.abs().sort_values(ascending=False)[:words_num_names_kept]
            return data[fe.index]
        else:
            return data

    def addressbook_gen(self, data, col='X_User_addressBook'):
        user_addressbook = data[col].map(lambda x: self.__read_addressbook(x))
        names_list = user_addressbook.dropna().map(lambda x: self.__name_extract(x))
        names_list = names_list.drop(names_list.loc[names_list.map(lambda x: len(x)) > self.max_names_num].index)
        return names_list

    def __read_addressbook(self, x):
        if x not in [None, np.nan, ''] and type(x) is not float:
            if type(json.loads(x)) is list:
                if len(json.loads(x)) > 0:
                    return json.loads(x)[0]['contents']
            elif type(json.loads(x)) is dict:
                return json.loads(x)['contents']
            else:
                return None
        else:
            return None

    def __words_counter(self, address, words):
        return sum(map(lambda x: 1 if words in x else 0, address))

    def __dict_gen(self, apply_data, name_length=20):
        # generate address dict from apply data
        addressbook = self.addressbook_gen(apply_data)
        names_list = []
        for name in addressbook:
            names_list.extend(name)
        names_series = list(filter(lambda x: len(x) <= name_length, names_list))
        names_dict = ''.join(names_series).replace('\'', '').replace(' ', '')
        return names_dict

    def __name_extract(self, x):
        names = []
        for d in x:
            if 'name' in d.keys():
                names.append(d['name'])
        return names

    def __words_col_gen(self, words_dict, names_map, n, freq=1, ratio=1):
        words = pd.DataFrame()
        dict_col_name = {}
        for i, name in enumerate(words_dict):
            if freq==1:
                words['word_{}_{}'.format(n,i)] = names_map.map(lambda x: self.__words_counter(x, name))
                dict_col_name['word_{}_{}'.format(n,i)] = name
            if ratio==1:
                words['word_ratio_{}_{}'.format(n,i)] = names_map.map(lambda x: self.__words_counter(x, name)/(len(x)+1))
                dict_col_name['word_ratio_{}_{}'.format(n,i)] = name
        words.index = names_map.index
        return words, dict_col_name


    def __num_addressbook_count(self, names):
        if len(names) > 0:
            return (pd.Series(names).map(lambda x: str.isdigit(x)) == True).sum() / len(names)
        else:
            return 0

    def __letter_addressbook_count(self, names):
        if len(names) > 0:
            return (pd.Series(names).map(lambda x: str.isalpha(x)) == True).sum() / len(names)
        else:
            return 0



class AddingFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, td=1, ylzc=1):
        self.td = td
        self.ylzc = ylzc
        self.td_features=[]
        self.ylzc_features=[]

    def fit(self, X, y=None):
        for col in X.columns:
            if 'applyLoanOnPlats' in col:
                self.td_features.append(col)
            elif 'YLZC_CDTB' in col and X[col].dtype in ['int', 'float']:
                self.ylzc_features.append(col)

        if self.ylzc == 1:
            self._ylzc_imputer = Imputer(strategy='most_frequent')
            self._ylzc_imputer.fit(X[self.ylzc_features])

        return self

    def transform(self, X, y=None):
        X_transformed = X.copy()
        lacked_cols = list(set(self.td_features)|set(self.ylzc_features) - set(X.columns))
        if len(lacked_cols)>0:
            X_transformed[lacked_cols] = 0
            
        X_transformed[self.td_features] = X_transformed[self.td_features].fillna(0)

        if self.ylzc == 1:
            X_transformed[self.ylzc_features] = self._ylzc_imputer.transform(X_transformed[self.ylzc_features])

        td_set = set(['applyLoanOnPlats30Cnt','applyLoanOnPlats12MonthCnt','applyLoanOnPlats6MonthCnt','applyLoanOnPlats3MonthCnt','applyLoanOnPlats7Cnt','applyLoanOnPlats60MonthCnt'])

        if self.td == 1:
            if len(td_set) == len(td_set & set(X_transformed.columns)):

                X_transformed['td_ratio_12_30'] = X_transformed['applyLoanOnPlats30Cnt'] / (X_transformed['applyLoanOnPlats12MonthCnt'] + 0.1)
                X_transformed['td_ratio_12_6'] = X_transformed['applyLoanOnPlats6MonthCnt'] / (X_transformed['applyLoanOnPlats12MonthCnt'] + 0.1)
                X_transformed['td_ratio_12_3'] = X_transformed['applyLoanOnPlats3MonthCnt'] / (X_transformed['applyLoanOnPlats12MonthCnt'] + 0.1)
                X_transformed['td_ratio_30_7'] = X_transformed['applyLoanOnPlats7Cnt'] / (X_transformed['applyLoanOnPlats30Cnt'] + 0.1)
                X_transformed['td_ratio_60_12'] = X_transformed['applyLoanOnPlats12MonthCnt'] / (X_transformed['applyLoanOnPlats60MonthCnt'] + 0.1)
                X_transformed['td_ratio+10_12_30'] = X_transformed['applyLoanOnPlats30Cnt'] / (X_transformed['applyLoanOnPlats12MonthCnt'] + 10)
                X_transformed['td_ratio+10_12_6'] = X_transformed['applyLoanOnPlats6MonthCnt'] / (X_transformed['applyLoanOnPlats12MonthCnt'] + 10)
                X_transformed['td_ratio+10_12_3'] = X_transformed['applyLoanOnPlats3MonthCnt'] / (X_transformed['applyLoanOnPlats12MonthCnt'] + 10)
                X_transformed['td_ratio+10_30_7'] = X_transformed['applyLoanOnPlats7Cnt'] / (X_transformed['applyLoanOnPlats30Cnt'] + 10)
                X_transformed['td_ratio+10_60_12'] = X_transformed['applyLoanOnPlats12MonthCnt'] / (X_transformed['applyLoanOnPlats60MonthCnt'] + 10)

        return X_transformed