import json
import pandas as pd
from mlx_jdx.dal import jdx_data
from mlx_database.mysql import MySql
from mlx_utility import config_manager as cm
from mlx_database.mongo import Mongo


class CategoryModelHelper:
    def __init__(self):
        # db client
        self.mysql_mlx_client = MySql(**cm.config['mysql_jd_cl'])
        self.mongo_mlx_client = Mongo(**cm.config['mongo_jd_cl'])

        # db access
        self.ml_result_access = jdx_data.MachineLearningJdResults(client=self.mysql_mlx_client)
        self.original_data_access = jdx_data.DerivativeProdData(client=self.mongo_mlx_client)
        self.threshold_access = jdx_data.PrThreshold(client=self.mysql_mlx_client)

    def get_input_data_df(self, app_id, fields=None):
        '''
        get user application data for model from mongo
        :param app_id: application id
        :param fields: attribute to be fetch from mongo
        :return: a dataframe made by 'fields' and the columns keep the sequence
        '''
        data_dict = self.original_data_access.get_data_by_appid_mongo_fields(app_id, fields)
        # list to store app value
        input_data_df = pd.DataFrame([data_dict])
        return input_data_df

    def save_ml_result(self, app_id, user_id, credit_ml, credit_final, result, threshold, judge_by, credit_by, model_name):
        '''
        save model decision result to database
        '''
        # self.original_data_access.save_credit_amount(app_id.upper(), credit_ml)
        return self.ml_result_access.save(app_id, user_id, credit_ml, credit_final, result, threshold, judge_by, credit_by, model_name)

    def get_threshold_bins_info(self, category_id, model):
        """get threshold from category and model
        """
        result = self.threshold_access.get_threshold_bins_by_category_and_model(category_id, model)
        thresholds, segment_bins = json.loads(result['thresholds']), json.loads(result['segment_bins'])
        return thresholds, segment_bins

    def update_mlcredit_by_appid_and_modelname(self, app_id, model_name, job_status, ml_credit):
        return self.ml_result_access.update_credit_ml_by_appid_and_modelname(app_id, model_name, job_status, ml_credit)
