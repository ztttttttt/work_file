# -*- coding: UTF-8 -*-
from pymongo import MongoClient
import logging


class Mongo(object):

    def __init__(self, host, user, pw, db, **kwargs):
        self._host = host
        self._user = user
        self._pw = pw
        self._db = db
        self._pool = None
        self._kwargs = kwargs
        self.get_conn()

    def _init_pool(self):
        uri = "mongodb://{0}:{1}@{2}".format(self._user, self._pw, self._host)
        self._kwargs.setdefault('readPreference', 'secondaryPreferred')
        self._kwargs.setdefault('maxPoolSize', 20)
        self._pool = MongoClient(host=uri, **self._kwargs)

    def get_conn(self):
        if not self._pool:
            self._init_pool()
        return self._pool

    def get_collection(self, collection, retry=3):
        try:
            return self.get_conn()[self._db][collection]
        except Exception as ex:
            logging.exception("mongo get_collection error!")
            retry -= 1
            if retry <= 0:
                raise ex
            else:
                self._init_pool()
                return self.get_collection(collection, retry)
