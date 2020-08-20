# -*- coding:utf-8 -*-
from mlo_database import dbdata
import numpy as np
import pymongo


class OmegaRuleCheckResultData(dbdata.DBData):
    def __init__(self, client=None, table='omega_ml_rule_check_result'):
        super(OmegaRuleCheckResultData, self).__init__(client, table)

    def save(self, app_id, rule):
        sql = "insert into {table_name}(app_id, user_id, hit_rule_code, actual_hit_rule_code, result, " \
              "actual_result, service, channel, group_tag) values('{app_id}', {user_id}, '{hit_rule_code}', " \
              "'{actual_hit_rule_code}',{result}, {actual_result}, '{service}', {channel}, '{group_tag}')"
        sql = sql.format(table_name=self.table, app_id=app_id,
                         user_id="'{}'".format(rule['user_id']) if rule['user_id'] else 'null',
                         hit_rule_code=rule['hit_rule'],
                         actual_hit_rule_code=rule['actual_hit_rule'], result=rule['result'],
                         actual_result=rule['actual_result'], service=rule['service'],
                         channel=rule['omega_channel'], group_tag=rule['group_tag'])
        return self.client.update(sql) == 1

    def get_all_data_by_date(self, date_start, date_end):
        sql = "select result from {table_name} where create_time >= '{date1}' and create_time < '{date2}'"
        sql = sql.format(table_name=self.table, date1=date_start, date2=date_end)
        return self.client.query(sql)

    def get_group_tag(self, user_id):
        sql = "select group_tag from {table_name} where user_id = '{user_id}'"
        sql = sql.format(table_name=self.table, user_id=user_id)
        return self.client.query(sql)

    def get_cg_count_by_date(self, start_time, end_time, group_tag):
        sql = "select COUNT(1) as cg_num from {table_name} where create_time >= '{start}' and create_time < '{end}' " \
              "and group_tag = '{group_tag}' and service='omega_open_card'"
        sql = sql.format(table_name=self.table, start=start_time, end=end_time, group_tag=group_tag)
        return self.client.query(sql)


class OmegaRulesData(dbdata.DBData):

    def __init__(self, client=None, table='omega_ml_rules'):
        super(OmegaRulesData, self).__init__(client, table)

    def get_rules_by_server(self, service_name, omega_channel):
        sql = "select rule_code, rule_name, pass_prob, prob_gen_manner, prob_params, check_model, model_params, " \
              "rule_level, convert_type_to from {table_name} where is_setup='1' and service = " \
              "'{service_name}' and channel = {omega_channel}".format(table_name=self.table, service_name=service_name,
                                                                      omega_channel=omega_channel)
        results = self.client.query(sql)
        return results if results else None

    def get_rule_by_rule_code(self, rule_code, omega_channel):
        sql = "select rule_code, rule_name, pass_prob, prob_gen_manner, prob_params, check_model, model_params " \
              "from {table_name} where is_setup='1' and rule_code='{rule_code}' and " \
              "channel = {omega_channel}".format(table_name=self.table, rule_code=rule_code, omega_channel=omega_channel)
        results = self.client.query(sql)
        return results if results else None


class DerivativeProdData(dbdata.DBData):
    def __init__(self, client=None, table='derivativevariables'):
        super(DerivativeProdData, self).__init__(client, table)
        self.key_name = '_id'

    def get(self, query, fields):
        if isinstance(fields, (tuple, list, np.ndarray)):
            fields = {x: 1 for x in fields}
        return self.client.get_collection(self.table).find_one(query, fields)

    def get_data_by_appid(self, app_id):
        query = {self.key_name: app_id.upper()}

        result = self.client.get_collection(self.table).find(query).next()
        result = {key.upper(): value for key, value in result.items()}

        return result

    def get_origin_data_by_appid(self, app_id):
        query = {self.key_name: app_id.upper()}
        result = self.client.get_collection(self.table).find(query).next()
        return result

    def update_by_app_id(self, app_id, data_dict, upsert=True):
        self.client.get_collection(self.table).update_one({self.key_name: app_id.upper()}, {'$set': data_dict}, upsert=upsert)

    def get_fields_by_appid(self, app_id, fields):
        return self.get({self.key_name: app_id.upper()}, fields)

    def get_userid_by_appid(self, app_id, user_id_name='Omega_User_Id'):
        return self.get({self.key_name: app_id.upper()}, [user_id_name])


class OmegaDeviceInfoData(dbdata.DBData):
    def __init__(self, client=None, table='deviceInfos'):
        super(OmegaDeviceInfoData, self).__init__(client, table)
        self.key_name = 'userId'

    def get(self, query, fields):
        if isinstance(fields, (tuple, list, np.ndarray)):
            fields = {x: 1 for x in fields}
        return self.client.get_collection(self.table).find_one(query, fields)

    def get_last_data_by_userid(self, user_id):
        query = {self.key_name: user_id.upper()}
        result = list(self.client.get_collection(self.table).find(query).sort('timestamp', pymongo.ASCENDING))
        return result[-1] if result else {}

    def get_fields_by_user_id(self, user_id, fields):
        return self.get({self.key_name: user_id.upper()}, fields)


class AddressBook(dbdata.DBData):
    def __init__(self, client=None, table='address_book'):
        super(AddressBook, self).__init__(client, table)
        self.key_name = 'userId'

    def get_address_book_by_userid(self, user_id):
        query = {self.key_name: user_id.upper()}
        result = list(self.client.get_collection(self.table).find(query).sort('create_time', pymongo.ASCENDING))
        return result[-1] if result else {}


class RiskPreauditData(dbdata.DBData):
    def __init__(self, client=None, table='preaudit_applications'):
        super(RiskPreauditData, self).__init__(client, table)

    def update_result_by_id(self, app_id, result):
        sql = "update {} set msg = '{}' where AppId = '{}'".format(self.table, result, app_id)
        return self.client.update(sql) == 1

    def get_data_by_appid(self, app_id):
        sql = "select * from {table_name} where appid = '{app_id}' ORDER BY create_time DESC".format(
            table_name=self.table, app_id=app_id)
        result = self.client.query(sql)
        return result[0] if result else {}


class OmegaModelResultsData(dbdata.DBData):
    def __init__(self, client=None, table='omega_ml_model_results'):
        super(OmegaModelResultsData, self).__init__(client, table)

    def save(self, app_id, raw_data):
        sql = "insert into {table_name}(app_id, result, threshold, model, is_pass, annotation, category_id) " \
              "values('{app_id}', {result}, {threshold}, '{model}', {is_pass}, {annotation}, '{category_id}')"
        sql = sql.format(table_name=self.table, app_id=app_id, result=raw_data['result'],
                         threshold=raw_data['threshold'], model=raw_data['model'], is_pass=raw_data['is_pass'],
                         annotation="'{}'".format(raw_data['annotation']) if raw_data['annotation'] else 'null',
                         category_id=raw_data['category_id'])
        return self.client.update(sql) == 1

    def save_credit(self, app_id, credit):
        sql = "update {} set credit_ml = {}, credit_by = 1 where app_id = '{}'".format(self.table, credit, app_id)
        return self.client.update(sql) == 1

    def get_score_model_by_appid(self, app_id):
        sql = "select result, model from {table_name} where app_id = '{app_id}'".format(table_name=self.table, app_id=app_id)
        result = self.client.query(sql)
        return (result[-1]['result'], result[-1]['model']) if result else (0, '')


class OmegaMlModelConfigs(dbdata.DBData):
    def __init__(self, client=None, table='omega_ml_model_configs'):
        super(OmegaMlModelConfigs, self).__init__(client, table)

    def get_all_model_configs(self):
        sql = "select * from {table_name} where is_setup=1".format(table_name=self.table)
        return self.client.query(sql)


class OmegaMlMultiModelResults(dbdata.DBData):
    def __init__(self, client=None, table='omega_ml_multi_model_results'):
        super(OmegaMlMultiModelResults, self).__init__(client, table)

    def save(self, model_name, score, threshold, app_id):
        sql = "insert into {table_name}(app_id, model_name, score, threshold) values('{app_id}','{model_name}', " \
              "{score}, {threshold})"
        sql = sql.format(table_name=self.table, app_id=app_id, score=score, threshold=threshold, model_name=model_name)
        return self.client.update(sql) == 1


class OmegaMlCreditConfigs(dbdata.DBData):
    def __init__(self, client=None, table='omega_ml_credit_configs'):
        super(OmegaMlCreditConfigs, self).__init__(client, table)

    def get_credit_configs(self):
        sql = "select * from {table_name} where is_setup=1".format(table_name=self.table)
        result = self.client.query(sql)
        return result


class OmegaCategoryRelations(dbdata.DBData):
    def __init__(self, client=None, table='omega_category_relations'):
        super(OmegaCategoryRelations, self).__init__(client, table)

    def read_category_relations(self):
        sql = "select category_id, relation from {}  " \
              "where relation != 'default' and delete_time is null".format(self.table)
        result = self.client.query(sql)
        return result

    def read_default_category_id(self):
        sql = "select category_id from {}  " \
              "where  relation = 'default' and delete_time is null".format(self.table)
        result = self.client.query_one(sql)
        return result if result else None
