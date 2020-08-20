import json
import pandas as pd
from asteval import Interpreter
import numpy as np
import logging
from mlx_jdx.dal import jdx_data
from mlx_database.mysql import MySql
from mlx_utility import config_manager as cm
from mlx_database.mongo import Mongo


class CategoryHelper:
    def __init__(self, mysql_client, mongo_client):

        # data access
        self.category_ID_access = jdx_data.CategoryRelation(client=mysql_client)

        self.original_app_data_access = jdx_data.DerivativeProdData(client=mongo_client)

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

    def determine_category(self, relation_list, app_data_dict, attr_keys_search_mongo, flag):
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

            if set(list(relation_dict.keys())) != set(attr_keys_search_mongo):
                continue

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
            res_out = self.category_ID_access.read_default_category_id(flag)
            return res_out['category_id']
