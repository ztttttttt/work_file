# -*- coding:utf-8 -
import json
import pickle
import logging
import numpy as np
import pandas as pd
from mlx_database.dbdata import DBData

class LtvMlModelConfigs(DBData):
    def __init__(self, client=None, table='ltv_ml_model_configs'):
        super(LtvMlModelConfigs, self).__init__(client, table)

    def get_all_model_configs(self):
        sql = "select * from {table_name} where is_setup=1".format(table_name=self.table)
        return self.client.query(sql)

class ModelInfo:
    def __init__(self, **model_config):
        self.category_id = model_config['category_id']
        self.model_name = model_config['model_name']
        self.threshold = model_config['model_threshold']
        self.prob = model_config['model_prob']
        estor_dk = self._load_model_from_file(model_config['model_path'])
        #new code
        self.estimator = estor_dk['model']
        self.fields = self.estimator.get_params()['enum'].clean_col_names
        self.estimator_name = estor_dk['model_name']
        self.estimator_train_date = estor_dk['train_date_score']['date']
        self.train_score = estor_dk['train_date_score']['score']
        self.data_label = estor_dk['data_label']
        self.credit_field = estor_dk['credit_field']
        self.principal_field = estor_dk['principal_field']


        self.model = estor_dk['model']
        self.positive_class_index = 1

    @staticmethod
    def _load_model_from_file(file_path):
        with open(file_path, 'rb') as f:
            pk_obj = pickle.load(f)
        return pk_obj

    def _model_predict_in_pipeline(self, input_data_df):

        pred_probas = self.model.predict_proba(input_data_df)
        positive_proba = pred_probas[:, self.positive_class_index][0]
        score = float(positive_proba)

        return score

    def predict(self, data_df):
        return self._model_predict_in_pipeline(data_df)

    def __repr__(self):
        return json.dumps({
            'prob': self.prob,
            'model_name': self.model_name,
            'threshold': self.threshold
        })

    def model_predict_in_pipeline(self, input_data_df):
        '''
        :param input_data:  two dimension array of features
        :param category_threshold:  scalar; category_threshold of category where this user is in
        :return: predict probability
        '''
        assert input_data_df.ndim == 2, 'input_data should be two dimension'

        # transform data and make prediction
        pred_probas = self.estimator.predict_proba(input_data_df)
        positive_proba = pred_probas[:, self.positive_class_index][0]
        return positive_proba



class ModelManager:
    """constructor

    Parameters
    ----------
    model_configs: list or iter-object, describe the model info, like:
                    [{
                        'category_id': '',
                        'model_name': '',
                        'model_prob': 0,
                        'model_path': ''
                    }]
    """

    def __init__(self, mysql_client):
        self.model_config = LtvMlModelConfigs(mysql_client)
        self._load_models()
        self._check_configs()

    def _load_models(self):
        model_configs = self.model_config.get_all_model_configs()
        self.category_models = {}
        for model_config in model_configs:
            self.category_models.setdefault("{}".format(model_config['category_id']), []).append(ModelInfo(**model_config))
        logging.info('load models over. {}'.format(self.category_models))

    def _check_configs(self):
        for k, v in self.category_models.items():
            if sum([record.prob for record in v]) != 1:
                logging.error("the sum of the probability must be one. category_target: {}".format(k))
                raise Exception("prob is wrong!")
        logging.info('check models done.')

    @staticmethod
    def _choose_one(models):
        prob_ = np.random.rand(1)[0]
        sum_tmp = 0
        for model in models:
            sum_tmp += model.prob
            if prob_ < sum_tmp:
                return model
        return None

    def predict(self, data_df, category_id):
        results = []
        for model_category_id, models in self.category_models.items():

            if str(model_category_id).lower() != str(category_id).lower():
                continue

            model = self._choose_one(models)
            score = model.predict(data_df)
            results.append(
                {
                    'category_id': model_category_id,
                    'threshold': model.threshold,
                    'model_name': model.model_name,
                    'score': score
                }
            )
        return results

    def all_predict(self, data_df, category_id):
        results = []
        for model_category_id, models in self.category_models.items():

            if str(model_category_id).lower() != str(category_id).lower():
                continue

            for model in models:
                score = model.predict(data_df)
                results.append(
                    {
                        'category_id': model_category_id,
                        'threshold': model.threshold,
                        'model_name': model.model_name,
                        'score': score
                    }
                )
        return results
