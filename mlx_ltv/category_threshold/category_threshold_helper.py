from collections import Counter
import json
import numpy as np
from asteval import Interpreter

from mlx_ltv.dal.category_access_data import (CategoryIDAccessData,
                                              ApplicationBriefAccessData,
                                              StatisticCounterAccessData,
                                              ThresholdAccessData,
                                              ThresholdParameterAccessData)
from mlx_ltv.dal.original_app_access_data import OriginalAppAccessData
import logging


class CategoryThresholdHelper:
    def __init__(self, mysql_client, mongo_client):
        # database client
        self.mysql_mlx_client = mysql_client
        self.mongo_mlx_client = mongo_client
        # data access
        self.category_ID_access = CategoryIDAccessData(client=self.mysql_mlx_client,
                                                       table='category_relations')
        self.app_brief_access = ApplicationBriefAccessData(client=self.mysql_mlx_client,
                                                           table='category_application_briefs')
        self.stat_counter_access = StatisticCounterAccessData(client=self.mysql_mlx_client,
                                                              table='category_statistic_counters')
        self.threshold_access = ThresholdAccessData(client=self.mysql_mlx_client,
                                                    table='category_model_thresholds')
        self.threshold_param_access = ThresholdParameterAccessData(client=self.mysql_mlx_client,
                                                                   table='category_update_parameters')
        self.original_app_data_access = OriginalAppAccessData(client=self.mongo_mlx_client,
                                                              table='derivativevariables')

    def get_app_dict_by_keys(self, app_id, keys):
        '''
        get application data according to the keys and these keys are SQL style
        :param app_id: application id
        :param keys: the attribute to search
        :return: a dict of that keys and their corresponding values
        '''
        relation_value_dict = self.original_app_data_access.read_app_data_by_SQL_fields(app_id, keys)
        return relation_value_dict

    def get_category_relations(self):
        '''
        get all activate relations
        :return: the list of all activate relations
        '''
        relation_list = self.category_ID_access.read_category_relations()
        return relation_list

    def determine_category(self, relation_list, app_data_dict):
        '''
        determine category by relation
        :param principal: the value user input
        :return: category_id
        '''
        aeval = Interpreter()
        category_id = None
        for rl in relation_list:
            agg_cmp = []
            relation_dict = json.loads(rl['relation'])  # convert the 'relation' json to dict
            for kk, vv in relation_dict.items():
                # evaluation the relation using the value of application
                if kk not in app_data_dict:
                    agg_cmp.append(False)
                else:
                    aeval.symtable['VALUE'] = app_data_dict[kk]
                    agg_cmp.append(aeval(vv))
            if np.array(agg_cmp).all():
                category_id = rl['category_id']
                break
        if category_id is not None:
            return category_id
        else:
            # cannot match the relation, just return the default one
            logging.warning('cannot match category, load default category_id')
            res_out = self.category_ID_access.read_default_category_id()
            return res_out['category_id']

    def create_app_brief_record(self, app_id, category_id):
        '''
        create a application brief record
        :param app_id:
        :param category_id:
        :return: True/False
        '''
        return self.app_brief_access.create_app_record(app_id, category_id)

    def update_app_counter(self, category_id):
        '''
        update the counter for the specific category
        :param category_id: category id
        :return: True/False
        '''
        return self.stat_counter_access.update_counter(category_id)

    def get_app_counter_stat_window(self, category_id):
        '''
        get counter and the update window
        :param category_id: category id
        :return: app_counter, statistic_window and pass_rate batch_size
        '''
        res_set = self.stat_counter_access.read_statistic_counter(category_id)
        return res_set['app_counter'], res_set['statistic_window'], res_set['pass_rate_sample_size']

    def calculate_current_pass_rate_update_model(self, category_id, batch_size):
        '''
        current pass rate according to 'window' application,and return model info of most frequently predict application
        :param category_id: category id
        :return: pass rate,model_info: model info; e.g. 'xgboost_2017-08-31_ispd7'
        '''
        res_set = self.app_brief_access.read_record_by_batch_size(category_id, batch_size)
        is_pass_arr = np.array([p['is_pass'] for p in res_set])
        model_info_arr = np.array([q['model_info'] for q in res_set if q['model_info'] != ''])

        curr_pass_rate = np.sum(is_pass_arr == 1) * 1.0 / len(is_pass_arr)
        # get the value of most frequent model info
        model_info = Counter(model_info_arr).most_common(1)[0][0]

        return curr_pass_rate, model_info

    def get_current_threshold(self, category_id, model_info):
        '''
        get current threshold
        :param category_id: category id
        :param model_info: the string to indicate the model
        :return: current threshold
        '''
        threshold_res = self.threshold_access.read_threshold(category_id, model_info)
        return threshold_res['threshold']

    def update_threshold(self, category_id, model_info, new_threshold, update_time):
        '''
        update category_threshold
        :param category_id: category id
        :param model_info: model info; e.g. 'xgboost_2017-08-31_ispd7'
        :param new_threshold: the calculated category_threshold
        :param update_time: update time, usually time now.
        :return:
        '''
        return self.threshold_access.update_threshold(category_id, model_info, new_threshold, update_time)

    def get_threshold_param(self, category_id):
        '''
        get parameter for updating category_threshold
        :param category_id: category id
        :param model_info: model info; e.g. 'xgboost_2017-08-31_ispd7'
        :return: parameter dict
        '''

        param_ = self.threshold_param_access.read_parameter(category_id)
        return param_

    def update_app_brief(self, app_id, model_info, is_pass, check_type):
        '''
        update application brief table
        :param app_id: UUID
        :param model_info: model info; e.g. 'xgboost_2017-08-31_ispd7'
        :param is_pass: whether this application passed or not
        :param check_type: decision point name
        :return:
        '''
        return self.app_brief_access.update_record(app_id, model_info, is_pass, check_type)

    #get info for channel
    def get_id_number(self, app_id, id_number_col):
        '''
        get id number from mongo
        :param app_id: application id
        :param id_number_col: the column name of id_number
        :return: id number
        '''
        res = self.original_app_data_access.read_app_data_by_mongo_fields(app_id, [id_number_col])
        return str(res[id_number_col])
