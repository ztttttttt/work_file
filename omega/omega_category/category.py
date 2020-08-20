# -*- coding:utf-8 -
from asteval import Interpreter
import json
import numpy as np
import logging

from omega.omega_dal.data_source import OmegaCategoryRelations
from omega.omega_dal.data_source import DerivativeProdData


class OmegaCategory(object):
    def __init__(self, mongo_client, mysql_client):
        self.deri_data = DerivativeProdData(mongo_client)
        self.cate_data = OmegaCategoryRelations(mysql_client)

    def get_category_relations(self):
        cate = self.cate_data.read_category_relations()
        return cate

    def get_default_category_relations(self):
        cate = self.cate_data.read_default_category_id()
        return cate

    def get_deri_dict_by_keys(self, app_id, keys):
        data = self.deri_data.get_fields_by_appid(app_id, keys)
        return data

    def get_category_by_relation(self, relation_list, app_data_dict):
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
            res_out = self.cate_data.read_default_category_id()
            return res_out['category_id']
