from mlx_database.dbdata import DBData
import numpy as np


class XData(DBData):
    def __init__(self, client, table="derivativevariables"):
        super().__init__(client, table)

    def get(self, query, fields):
        if isinstance(fields, (tuple, list, np.ndarray)):
            fields = {x: 1 for x in fields}
        return self.client.get_collection(self.table).find_one(query, fields)

    def get_many(self, query, fields):
        if isinstance(fields, (tuple, list, np.ndarray)):
            fields = {x: 1 for x in fields}
        return list(self.client.get_collection(self.table).find(query, fields))

    def get_by_app_id(self, app_id, fields):
        app_id = str(app_id).upper()
        return self.get({'_id': app_id}, fields)

    def update_one(self, query, data_dict, upsert=True):
        self.client.get_collection(self.table).update_one(query, {'$set': data_dict}, upsert=upsert)

    def update_by_app_id(self, app_id, data_dict, upsert=True):
        app_id = str(app_id).upper()
        self.client.get_collection(self.table).update_one({'_id': app_id}, {'$set': data_dict}, upsert=upsert)

    def insert_one(self, data_dict):
        self.client.get_collection(self.table).insert_one(data_dict)
