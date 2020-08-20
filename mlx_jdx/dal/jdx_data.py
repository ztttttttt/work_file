import json
import numpy as np
import pymongo
from mlx_database.dbdata import DBData
from mlx_database import mysql, mongo


class JDXData(DBData):
    def __init__(self, client, table="derivables"):
        super().__init__(client, table)

    def get(self, query, fields):
        if isinstance(fields, (tuple, list, np.ndarray)):
            fields = {x: 1 for x in fields}
        return self.client.get_collection(self.table).find_one(query, fields)

    def get_lastone(self, query, fields):
        if isinstance(fields, (tuple, list, np.ndarray)):
            fields = {x: 1 for x in fields}
        result = list(self.client.get_collection(self.table).find(query, fields).sort('create_time', pymongo.ASCENDING))
        return result[-1] if result else None

    def get_many(self, query, fields):
        if isinstance(fields, (tuple, list, np.ndarray)):
            fields = {x: 1 for x in fields}
        return list(self.client.get_collection(self.table).find(query, fields))

    def get_by_app_id(self, app_id, fields):
        return self.get({'_id': app_id}, fields)

    def update_one(self, query, data_dict, upsert=True):
        self.client.get_collection(self.table).update_one(query, {'$set': data_dict}, upsert=upsert)

    def update_by_app_id(self, app_id, data_dict, upsert=True):
        self.client.get_collection(self.table).update_one({'_id': app_id}, {'$set': data_dict}, upsert=upsert)

    def insert_one(self, data_dict):
        self.client.get_collection(self.table).insert_one(data_dict)

    def get_last_one_by_mobile(self, mobile, fields):
        return self.get_lastone({'X_Mobile': mobile}, fields)


class MLWithdrawModelResults(DBData):
    def __init__(self, client):
        super(MLWithdrawModelResults, self).__init__(table='jdx_withdraw_model_results',
                                                     client=client)

    def save(self, app_id, user_id, category_id, result, threshold, model):
        sql = "insert into {} " \
              "(app_id, user_id, category_id, result, threshold, model) " \
              "values ('{}', '{}', '{}', {}, {},'{}')".format(
            self.table, app_id, user_id, category_id, result, threshold, model)

        return self.client.update(sql) == 1


class MachineLearningJdResults(DBData):
    def __init__(self, client):
        super(MachineLearningJdResults, self).__init__(table='machine_learning_jdcl_results',client=client)
        self.key_name = "app_id"
        self.date_key = "create_time"

    def save(self, app_id, user_id, credit_ml, credit_final, result, threshold, judge_by, credit_by, model_name,
             category_id):
        sql = "insert into {} " \
              "(app_id, user_id, credit_ml, credit_final, result, threshold, judge_by, credit_by, model, category_id) " \
              "values ('{}', '{}', {}, {}, '{}', '{}', {}, {}, '{}', '{}')".format(self.table, app_id, user_id,
                                                                                   credit_ml,
                                                                                   credit_final, json.dumps(result),
                                                                                   threshold, judge_by,
                                                                                   credit_by, model_name, category_id)
        return self.client.update(sql) == 1

    def update_credit_ml_by_appid_and_modelname(self, app_id, model_name, job_status, credit_ml):
        sql = "update {} set credit_ml = {} where app_id = '{}' and model='{}' and judge_by={}".format(self.table,
                                                                                                       credit_ml,
                                                                                                       app_id,
                                                                                                       model_name,
                                                                                                       job_status)
        return self.client.update(sql) > 0


class JdxOPScoreResults(DBData):
    def __init__(self, client):
        super(JdxOPScoreResults, self).__init__(table='jdx_op_score_results',client=client)

    def save(self, app_id, user_id, target, server, category_id, model, model_score, op_score):
        sql = "insert into {} " \
              "(app_id, user_id, target, server, category_id, model, model_score, op_score) " \
              "values ('{}', '{}', '{}', '{}', '{}', '{}', {}, {})".format(self.table, app_id, user_id, target, server, category_id, model, model_score, op_score)
        return self.client.update(sql) == 1


class DerivativeProdData(DBData):
    def __init__(self, client=None, config=None):
        super(DerivativeProdData, self).__init__(table='derivables', client=client)
        self.key_name = '_id'
        self.sqldb_to_mongo_attr_mapping = {
            'source': 'X_APP_Source'
        }
        self.mongo_to_sqldb_attr_mapping = {v: k for k, v in self.sqldb_to_mongo_attr_mapping.items()}

    def get_data_by_appid(self, app_id):
        query = {self.key_name: app_id.upper()}
        result = {}
        try:
            result = self.client.get_collection(self.table).find(query).next()
            result = {key: value for key, value in result.items()}
        except Exception:
            pass
        return result

    def get_data_by_appid_mongo_fields(self, app_id, fields):
        query = {self.key_name: app_id.upper()}
        if isinstance(fields, (tuple, list, np.ndarray)):
            fields = {x: 1 for x in fields}
        result = {}
        try:
            result = self.client.get_collection(self.table).find_one(query, fields)
        except Exception:
            pass
        return result

    def read_app_data_by_SQL_fields(self, app_id, fields):
        query = {self.key_name: app_id.upper()}
        if isinstance(fields, (tuple, list, np.ndarray)):
            fields = {x: 1 for x in fields}
        result = {}
        try:
            # convert field according to the mapping dict
            fields_ts = self.__attr_map(fields, self.sqldb_to_mongo_attr_mapping)
            result = self.client.get_collection(self.table).find_one(query, fields_ts)
            # return mongo data use SQL keys
            result = self.__attr_map(result, self.mongo_to_sqldb_attr_mapping)
        except Exception:
            pass
        return result

    def __attr_map(self, input_dict, map_dict):
        if not input_dict:
            return {}
        return {(map_dict[k] if k in map_dict else k): v for k, v in input_dict.items()}

    def save_credit_amount(self, app_id, credit_amount, upsert=True, prefix=""):
        data_dict = {"X_ML_CreditAmount" + prefix: credit_amount}
        self.client.get_collection(self.table).update_one({self.key_name: app_id}, {'$set': data_dict}, upsert=upsert)


class RuleCheckResultsData(DBData):
    def __init__(self, client=None, config=None):
        super(RuleCheckResultsData, self).__init__(table="jd_ml_rule_check_result", client=client)

    def save(self, app_id, rule):
        sql = "insert into {table_name}(app_id, category_id, hit_rule_code, actual_hit_rule_code, result, actual_result, user_id, group_tag, rule_type) " \
              "values('{app_id}', '{category_id}', '{hit_rule_code}', '{actual_hit_rule_code}',{result}, {actual_result}, '{user_id}', '{group_tag}', '{rule_type}')"
        sql = sql.format(table_name=self.table, app_id=app_id, category_id=rule['category_id'], hit_rule_code=rule['hit_rule'],
                         actual_hit_rule_code=rule['actual_hit_rule'], result=rule['result'],
                         actual_result=rule['actual_result'], user_id=rule['user_id'], group_tag=rule['group_tag'],
                         rule_type=rule['rule_type'])
        return self.client.update(sql) == 1

    def get_cg_count_by_date_tag(self, start_date, end_date, group_tag):
        sql = "select count(distinct user_id) as cnt from {table_name} " \
              "where create_time between '{start_date}' and '{end_date}' " \
              "and group_tag='{group_tag}' and rule_type='c'"
        sql = sql.format(table_name=self.table, start_date=start_date, end_date=end_date, group_tag=group_tag)
        results = self.client.query(sql)
        return results[0]['cnt'] if results else 0

    def get_open_card_rule_check_result(self, user_id):
        sql = "select hit_rule_code, actual_hit_rule_code from {table_name} " \
              "where user_id = '{user_id}' and rule_type='c' order by create_time asc"
        sql = sql.format(table_name=self.table, user_id=user_id)
        results = self.client.query(sql)
        return results[-1] if results else None

    def get_group_tag_by_user_id(self, user_id):
        sql = "select group_tag from {table_name} " \
              "where user_id = '{user_id}' order by create_time desc"
        sql = sql.format(table_name=self.table, user_id=user_id)
        results = self.client.query(sql)
        return results[-1]['group_tag'] if results else None

    def update_by_req_id(self, req_id, app_id, user_id):
        sql = "update {table_name} set app_id='{app_id}', user_id='{user_id}' where app_id='{req_id}'"
        sql = sql.format(table_name=self.table, app_id=app_id, user_id=user_id, req_id=req_id)
        ret = self.client.update(sql) >= 0
        return ret


class JDRulesData(DBData):

    def __init__(self, client=None, config=None):
        super(JDRulesData, self).__init__(table="jd_ml_rules", client=client)

    def get_rules_by_server(self, server_name, rule_type, category_id):
        sql = "select rule_code, rule_name, pass_prob, prob_gen_manner, prob_params, check_model, model_params, " \
              "rule_level from {table_name} where is_setup='1' and category_id='{category_id}' and server = " \
              "'{server_name}' and rule_type='{rule_type}'".format(table_name=self.table, category_id=category_id,
                                                                   server_name=server_name,rule_type=rule_type)
        results = self.client.query(sql)
        return results if results else None

    def get_rule_by_rule_code(self, rule_code, rule_type, category_id):
        sql = "select rule_code, rule_name, pass_prob, prob_gen_manner, prob_params, check_model, model_params " \
              "from {table_name} where is_setup='1' and category_id='{category_id}' and rule_code='{rule_code}' " \
              "and rule_type='{rule_type}'".format(table_name=self.table, category_id=category_id,
                                                   rule_code=rule_code, rule_type=rule_type)
        results = self.client.query(sql)
        return results if results else None


class PrThreshold(DBData):
    def __init__(self, client=None, config=None):
        super(PrThreshold, self).__init__(table="jdx_pass_rate_thresholds", client=client)

    def get_threshold_bins_by_category_and_model(self, category_id, model):
        stmt = "select thresholds, segment_bins from {} where category_id='{}' and model='{}'"
        stmt = stmt.format(self.table, category_id, model)
        results = self.client.query(stmt)
        return results[0] if results else None


class CreditAmountStragetyResult(DBData):
    def __init__(self, client=None, config=None):
        super(CreditAmountStragetyResult, self).__init__(table="jdx_credit_strategy_result", client=client)

    # def save_batch_result(self, appid, judge_strategy, credit_results):
    #     strategy_cnt = len(credit_results)
    #     is_judged = [0] * strategy_cnt
    #     strategies = list(credit_results.keys())
    #     amounts = [credit_results[strategy] for strategy in strategies]
    #     is_judged[strategies.index(judge_strategy)] = 1
    #     appids = [appid] * strategy_cnt
    #     stmt = "insert into {} (strategy_name, credit_amount, app_id, is_judged) values {}"
    #     stmt = stmt.format(self.table, ','.join(["{}".format(record) for record in zip(strategies, amounts, appids, is_judged)]))
    #     return self.client.update(stmt) > 0
    def save(self, appid, credit_amount, strategy_name, is_judged, thresholds, segment_bins):
        stmt = "insert into {} (strategy_name, credit_amount, app_id, is_judged, thresholds, segment_bins) values ('{}', {}, '{}', {}, '{}', '{}')".format(
            self.table, strategy_name, credit_amount, appid, is_judged, thresholds, segment_bins)
        return self.client.update(stmt) > 0


class CategoryRelation(DBData):
    def __init__(self, client=None, config=None):
        super(CategoryRelation, self).__init__(table='jdx_category_relations', client=client)

    def read_category_relations(self):
        sql = "select category_id, relation from {}  " \
              "where relation != 'default' and relation != 'default_withdraw' and delete_time is null".format(
            self.table)
        result = self.client.query(sql)
        return result

    def read_default_category_id(self, flag):
        if flag == 'open_card':
            sql = "select category_id from {}  " \
                  "where  relation = 'default' and delete_time is null".format(self.table)
        elif flag == 'withdraw':
            sql = "select category_id from {}  " \
                  "where  relation = 'default_withdraw' and delete_time is null".format(self.table)
        else:
            raise ValueError('get unknown flag:{}'.format(flag))
        result = self.client.query_one(sql)
        return result if result else None


class CategoryModelRelation(DBData):
    def __init__(self, client=None, config=None):
        super(CategoryModelRelation, self).__init__(table='jdx_category_model_relations', client=client)

    def get_all_relations_by_prob(self, prob_thres=0):
        sql = "select category_id, model_name, target, estor_path, prob, op_coef, op_intercept from {} " \
              "where prob > {}".format(self.table, prob_thres)
        result = self.client.query(sql)
        return result if result else None


class RecreditRecord(DBData):
    def __init__(self, client=None, config=None):
        super(RecreditRecord, self).__init__(table='jdx_recredit_record', client=client)

    def insert(self, app_id, user_id, auth_type, recredit_amount, recredit_rule):
        sql = "insert into {} (app_id, user_id, auth_type, recredit_amount, recredit_rule) " \
              "values ('{}', '{}', '{}', {}, '{}')".format(self.table, app_id, user_id, auth_type, recredit_amount,
                                                           recredit_rule)
        return self.client.update(sql) > 0
