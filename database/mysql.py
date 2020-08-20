# -*- coding: UTF-8 -*-
from sqlalchemy import create_engine
import logging
from urllib import parse


class MySql(object):
    def __init__(self, host, port, user, pw, db, pool_size=20, pool_timeout=60, pool_recycle=3600, **kwargs):
        self._host = host
        self._port = port
        self._user = user
        self._pw = pw
        self._db = db
        self._pool = None
        self._pool_size = pool_size
        self._pool_timeout = pool_timeout
        self._pool_recycle = pool_recycle
        self._kwargs = kwargs
        self._init_pool()

    def _init_pool(self):
        config = "mysql+mysqldb://{user}:{pw}@{host}:{port}/{db}?charset=utf8".format(user=self._user,
                                                                                      pw=parse.quote_plus(self._pw),
                                                                                      host=self._host, port=self._port,
                                                                                      db=self._db)
        self._pool = create_engine(config, pool_size=self._pool_size, pool_timeout=self._pool_timeout,
                                   pool_recycle=self._pool_recycle, **self._kwargs)

    def get_conn(self):
        try:
            return self._pool.connect()
        except Exception as ex:
            logging.exception("mysql get_conn error!")
            self._init_pool()
            return self._pool.connect()

    def query(self, sql, args=None, retry=3):
        try:
            conn = self.get_conn()
            result = conn.execute(sql)
            result_dict_list = []
            for r in result:
                result_dict_list.append(dict(r))
            return result_dict_list
        except Exception as ex:
            logging.exception("mysql query error!")
            retry -= 1
            if retry <= 0:
                raise ex
            else:
                self._init_pool()
                return self.query(sql, args, retry)

    def query_one(self, sql, args=None, retry=3):
        try:
            conn = self.get_conn()
            result = conn.execute(sql)
            one = result.fetchone()
            return dict(one) if one else None
        except Exception as ex:
            logging.exception("mysql query_one error!")
            retry -= 1
            if retry <= 0:
                raise ex
            else:
                self._init_pool()
                return self.query_one(sql, args, retry)

    def update(self, sql, args=None, retry=3):
        try:
            conn = self.get_conn()
            result = conn.execute(sql)
            return result.rowcount
        except Exception as ex:
            logging.exception("mysql update error!")
            retry -= 1
            if retry <= 0:
                raise ex
            else:
                self._init_pool()
                return self.update(sql, args, retry)
