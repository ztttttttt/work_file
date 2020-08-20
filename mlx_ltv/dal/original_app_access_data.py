import numpy as np
from mlx_database.dbdata import DBData


class OriginalAppAccessData(DBData):
    def __init__(self, client=None, table=None):
        super(OriginalAppAccessData, self).__init__(client, table)
        self.sqldb_to_mongo_attr_mapping = {
            'principal': 'X_APP_Principal',
            'repayments': 'X_APP_Repayments',
            'user_repayment_times': 'X_USER_RepaymentTimes'
        }
        self.mongo_to_sqldb_attr_mapping = {v: k for k, v in self.sqldb_to_mongo_attr_mapping.items()}

    def read_app_data_by_mongo_fields(self, app_id, fields):
        if isinstance(fields, (tuple, list, np.ndarray)):
            fields = {x: 1 for x in fields}
        app_id = str(app_id).upper()
        query = {'_id': app_id}
        res_set = self.client.get_collection(self.table).find_one(query, fields)
        return res_set

    def read_app_data_by_SQL_fields(self, app_id, fields):
        if isinstance(fields, (tuple, list, np.ndarray)):
            fields = {x: 1 for x in fields}
        app_id = str(app_id).upper()
        query = {'_id': app_id}
        # convert field according to the mapping dict
        fields_ts = self.__attr_map(fields, self.sqldb_to_mongo_attr_mapping)
        res_set = self.client.get_collection(self.table).find_one(query, fields_ts)

        # return mongo data use SQL keys
        return self.__attr_map(res_set, self.mongo_to_sqldb_attr_mapping)

    def __attr_map(self, input_dict, map_dict):
        if not input_dict:
            return {}
        return {(map_dict[k] if k in map_dict else k): v for k, v in input_dict.items()}
