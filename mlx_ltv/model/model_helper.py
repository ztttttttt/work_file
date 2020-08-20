import pandas as pd

from mlx_ltv.dal.category_access_data import ThresholdAccessData
from mlx_ltv.dal.ml_result_access_data import MLResultAccessData
from mlx_ltv.dal.original_app_access_data import OriginalAppAccessData
from mlx_ltv.dal.ml_available_credit_access import MLAvailableCreditAccessData
from mlx_database.mysql import MySql
from mlx_utility import config_manager as cm
from mlx_database.mongo import Mongo


class CategoryModelHelper:
    def __init__(self, mysql_client, mongo_client):
        # database client
        self.mysql_mlx_client = mysql_client
        self.mongo_mlx_client = mongo_client

        # db access
        self.threshold_access = ThresholdAccessData(client=self.mysql_mlx_client,
                                                    table='category_model_thresholds')
        self.ml_result_access = MLResultAccessData(client=self.mysql_mlx_client,
                                                   table='machine_learning_risk_model_results')

        self.ml_available_credit = MLAvailableCreditAccessData(client=self.mysql_mlx_client,
                                                               table='machine_learning_max_available_credits')

        self.original_data_access = OriginalAppAccessData(client=self.mongo_mlx_client,
                                                          table='derivativevariables')

    def get_input_data_df(self, app_id, fields):
        '''
        get user application data for model from mongo
        :param app_id: application id
        :param fields: attribute to be fetch from mongo
        :return: a dataframe made by 'fields' and the columns keep the sequence
        '''
        data_dict = self.original_data_access.read_app_data_by_mongo_fields(app_id, fields)
        # list to store app value
        agg_attr = []
        for attr in fields:
            agg_attr.append(data_dict.get(attr))
        # use pandas to format data, the dimension should be 1 x len(fields)
        input_data_df = pd.DataFrame(agg_attr, index=fields).T
        return input_data_df

    def get_current_threshold(self, category_id, model_info):
        '''
        get category id threshold
        :param category_id: category id
        :param model_info: the string indicate model information
        :return: current threshold
        '''
        threshold_res = self.threshold_access.read_threshold(category_id, model_info)
        return threshold_res['threshold']

    def save_ml_result(self, user_id, app_id, category_id, result, threshold, model_info):
        '''
        save model decision result to database
        :param user_id: user id
        :param app_id: application id
        :param category_id: category id to define this application
        :param result: model prediction probability
        :param threshold: the threshold when making decision
        :param model_info: the string indicate the model
        :return: True/False
        '''
        return self.ml_result_access.save_result(user_id, app_id, category_id, result, threshold, model_info)

    def save_credit(self, user_id, app_id, category_id, max_credit):
        '''
        save maximum available credit
        :return: True/False
        '''
        return self.ml_available_credit.save_credit(user_id, app_id, category_id, max_credit)
