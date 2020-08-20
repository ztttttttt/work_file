import logging
import json
import pickle
import pandas as pd
import numpy as np
import math

from mlx_jdx.data_integration.woe_mapper import Tansform
from mlx_jdx.model_online.risk_model import RiskModel
from mlx_jdx.dal import jdx_data
from mlx_database.mysql import MySql
from mlx_utility import config_manager as cm
from mlx_database.mongo import Mongo


class ModelHelper:
    def __init__(self, mysql_client, mongo_client):
        # db access
        self.ml_withdraw_result_access = jdx_data.MLWithdrawModelResults(client=mysql_client)
        self.ml_result_access = jdx_data.MachineLearningJdResults(client=mysql_client)
        self.op_socre_result_access = jdx_data.JdxOPScoreResults(client=mysql_client)
        self.original_data_access = jdx_data.DerivativeProdData(client=mongo_client)
        self.threshold_access = jdx_data.PrThreshold(client=mysql_client)

    def get_input_data_df(self, app_id, fields):
        '''
        get user application data for model from mongo
        :param app_id: application id
        :param fields: attribute to be fetch from mongo
        :return: a dataframe made by 'fields' and the columns keep the sequence
        '''
        data_dict = self.original_data_access.get_data_by_appid_mongo_fields(app_id, fields)
        # list to store app value
        agg_attr = []
        for attr in fields:
            agg_attr.append(data_dict.get(attr))
        # use pandas to format data, the dimension should be 1 x len(fields)
        input_data_df = pd.DataFrame(agg_attr, index=fields).T
        return input_data_df

    def get_datadict_by_appid(self, app_id):
        data_dict = self.original_data_access.get_data_by_appid(app_id)
        return data_dict

    def save_ml_result(self, app_id, user_id, credit_ml, credit_final, result, threshold, judge_by, credit_by,
                       model_name, category_id):
        '''
        save model decision result to database
        '''
        self.original_data_access.save_credit_amount(app_id.upper(), credit_ml)
        return self.ml_result_access.save(app_id, user_id, credit_ml, credit_final, result, threshold, judge_by,
                                          credit_by, model_name, category_id)

    def save_op_socre_result(self, app_id, user_id, target, server, category_id, model, model_score, op_score):
        self.op_socre_result_access.save(app_id, user_id, target, server, category_id, model, model_score, op_score)

    def get_threshold_bins_info(self, category_id, model):
        """get threshold from category and model
        """
        result = self.threshold_access.get_threshold_bins_by_category_and_model(category_id, model)
        thresholds, segment_bins = json.loads(result['thresholds']), json.loads(result['segment_bins'])
        return thresholds, segment_bins

    def save_withdraw_model_result(self, app_id, user_id, category_id, result, threshold, model):
        return self.ml_withdraw_result_access.save(app_id, user_id, category_id, result, threshold, model)


class ModelInfo:
    def __init__(self, **model_config):
        self.category_id = model_config['category_id']
        self.model_name = model_config['model_name']
        self.target = model_config['target']
        self.prob = model_config['prob']
        self.model_path = model_config['estor_path']
        self.op_coef = model_config['op_coef']
        self.op_intercept = model_config['op_intercept']
        estor_dk = self._load_model_from_file(model_config['estor_path'])
        try:
            self.model = estor_dk['model']
        except:
            self.model = estor_dk
        try:
            self.fields = self.model.get_params()['enum'].clean_col_names
        except:
            self.fields = self.model.model_features

        try:
            self.model_info = estor_dk['model_info']
        except:
            self.model_info = 'xgboost_2018-11-01_2019-02-02_rand_train_fs30'
        self.positive_class_index = 1

    def _load_model_from_file(self, file_path):
        with open(file_path, 'rb') as f:
            pk_obj = pickle.load(f)
        return pk_obj

    def _model_predict_in_pipeline(self, input_data_df):
        '''
        :param input_data:  two dimension array of features
        :param category_threshold:  scalar; category_threshold of category where this user is in
        :return: predict probability
        '''
        assert input_data_df.ndim == 2, 'input_data should be two dimension'

        # transform data and make prediction
        pred_probas = self.model.predict_proba(input_data_df)
        positive_proba = pred_probas[:, self.positive_class_index][0]
        return float(positive_proba)

    def _woe_model_predict_in_pipeline(self, input_data_df):    # 临时的
        '''
        :param input_data:  two dimension array of features
        :param category_threshold:  scalar; category_threshold of category where this user is in
        :return: predict probability
        '''
        assert input_data_df.ndim == 2, 'input_data should be two dimension'

        # transform data and make prediction
        pred_probas = self.model.predict_proba(input_data_df)
        positive_proba = pred_probas[:, 0][0]
        return float(positive_proba)

    def _filter_by_fields(self, data_dict):
        agg_attr = []
        for attr in self.fields:
            agg_attr.append(data_dict.get(attr))
        # use pandas to format data, the dimension should be 1 x len(fields)
        result_df = pd.DataFrame(agg_attr, index=self.fields).T
        return result_df

    def woe_tansform(self, data_dict):
        woe_origin_vars_path = self.model_path.split('model.pkl')[0] + 'woe_origin_vars.pkl'
        vars_first_path = self.model_path.split('model.pkl')[0] + 'vars_first.pkl'
        woe_encoder_path = self.model_path.split('model.pkl')[0] + 'woe_encoder.pkl'
        with open(woe_origin_vars_path, 'rb') as f:
            woe_origin_vars = pickle.load(f)
        with open(vars_first_path, 'rb') as f:
            vars_first = pickle.load(f)
        with open(woe_encoder_path, 'rb') as f:
            woe_encoder = pickle.load(f)
        woeTansform = Tansform(woe_origin_vars, vars_first, woe_encoder)
        model_data_df = woeTansform.tansform(self.fields, data_dict)
        return model_data_df

    def predict(self, data_dict):
        try:
            model_data = self.woe_tansform(data_dict)
            model_socre = self._woe_model_predict_in_pipeline(model_data)
        except Exception as e:
            model_data = self._filter_by_fields(data_dict)
            model_socre = self._model_predict_in_pipeline(model_data)
        return model_socre

    def __repr__(self):
        return json.dumps({
            'target': self.target,
            'prob': self.prob,
            'model_name': self.model_name,
            'model_info': self.model_info,
            'op_coef': self.op_coef,
            'op_intercept': self.op_intercept
        })


class ModelManager:
    """constructor

    Parameters
    ----------
    model_configs: list or iter-object, describe the model info, like:
                    {
                        'category_id': '',
                        'model_name': '',
                        'target': '',
                        'prob': 0,
                        'estor_path': ''
                    }
    """

    def __init__(self, mysql_client):
        self.model_category_access = jdx_data.CategoryModelRelation(client=mysql_client)
        self._load_models()
        self._check_configs()

    def _load_models(self):
        model_configs = self.model_category_access.get_all_relations_by_prob()
        self.category_models = {}
        for model_config in model_configs:
            self.category_models.setdefault("{}_{}".format(model_config['category_id'], model_config['target']),[]).append(ModelInfo(**model_config))
        logging.info('load models over. {}'.format(self.category_models))

    def _check_configs(self):
        for k, v in self.category_models.items():
            if sum([record.prob for record in v]) != 1:
                logging.error("the sum of the probability must be one. category_target: {}".format(k))
                raise Exception("prob is wrong!")
        logging.info('check models done.')

    def _choose_one(self, models):
        choice_i = np.random.choice(1000)
        sum_tmp = 0
        for model in models:
            sum_tmp += model.prob * 1000
            if choice_i < sum_tmp:
                return model
        return None

    def predict(self, data_dict, category_id):
        results = []
        for category_target, models in self.category_models.items():
            model = self._choose_one(models)
            score = model.predict(data_dict)
            op_score = self.get_op_score(score, model.op_coef, model.op_intercept)
            modle_category_id, target = category_target.split('_')
            if modle_category_id != category_id:
                continue
            results.append(
                {
                    'category_id': modle_category_id,
                    'target': target,
                    'model_name': model.model_name,
                    'score': score,
                    'op_score': op_score
                }
            )
        return results

    def get_op_score(self, model_score, op_coef, op_intercept):
        if not (op_coef and op_intercept):
            return 0
        model_ln_odds = math.log(model_score / (1 - model_score)) / math.log(2)
        ln_real_odds = op_coef * model_ln_odds + op_intercept
        op_score = max(min(60 * ln_real_odds + 300, 1000), 0)
        return op_score
