from pymongo import MongoClient
import pandas as pd

from mlx_database.hive import Hive
from mlx_database.mysql import MySql


class DBLoader:
    def __init__(self):
        self.hive_client = Hive(**{
            'host': "10.31.52.73",
            'port': 10000,
            'user': "wangli",
            'pw': 'OuniV7qc41vEwNfP'
        })

        self.mongo_jd_client = MongoClient(
            "mongodb://{user_name}:{pw}@{host}:{port}/{db}".format(
                user_name='mxjd', pw='Nko5jJ4gfiM3Eq7lRprY',
                host='10.47.89.174', port=27028, db='prod_jdcl'))

    def read_hive(self, sql):
        result = self.hive_client.query(sql)
        return result

    def _query_mongo(self, query, fields=None):
        if fields is not None:
            data = self.mongo_jd_client['prod_jdcl']['derivables'].find(query, fields)
        else:
            data = self.mongo_jd_client['prod_jdcl']['derivables'].find(query, batch_size=1000)
        return data

    def get_df(self, app_ids, columns=None, exclude=None):
        query = {'_id': {"$in": app_ids}}
        if exclude:
            res = self._query_mongo(query, exclude)
            return pd.DataFrame(list(res))
        if columns is not None:
            fields = {col: 1 for col in columns}
            res = self._query_mongo(query, fields)
        else:
            res = self._query_mongo(query)
        df = pd.DataFrame(list(res))
        return df

    def read_mysql(self, sql):
        mysql_client = MySql(host='10.47.89.174',
                             port=3229,
                             user='devml',
                             pw='gGynbdutOr3Eeg0I',
                             db='maple_leaf')
        res = mysql_client.query(sql=sql)
        return pd.DataFrame(res)
