from mlx_database.dbdata import DBData


class RuleData(DBData):
    def __init__(self, client, table=None):
        super().__init__(client, table)

    def get_rule_collection_results_by_app_id(self, app_id):
        sql = "select rule_collection_id, rule_collection_version, result, first_hit_rule_code " \
              "from app_rule_collection_results where app_id = '{}'".format(app_id)
        return self.client.query(sql)

    def get_rule_collection_result_by_app_id_and_collection_id(self, app_id, rule_collection_id):
        sql = "select rule_collection_version, result, first_hit_rule_code " \
              "from app_rule_collection_results " \
              "where app_id = '{}' and rule_collection_id = '{}'".format(app_id, rule_collection_id)
        return self.client.query(sql)

    def save_rule_result(self, app_id, rule_code, result, result_actual, rule_type, rule_collection_id):
        sql = "insert into app_rule_results " \
              "(app_id, rule_code, result, result_actual, rule_type, rule_collection_id) " \
              "values ('{}', '{}', {}, {}, '{}', '{}')".format(
            app_id, rule_code, result, result_actual, rule_type, rule_collection_id)
        return self.client.update(sql) == 1

    def save_rule_collection_result(self, app_id, rule_collection_id, rule_collection_version, result,
                                    first_hit_rule_code):
        if first_hit_rule_code is None:
            first_hit_rule_code = ""
        sql = "insert into app_rule_collection_results " \
              "(app_id, rule_collection_id, rule_collection_version, result, first_hit_rule_code) " \
              "values ('{}', '{}', '{}', {}, '{}')".format(app_id, rule_collection_id, rule_collection_version, result,
                                                           first_hit_rule_code)
        return self.client.update(sql) == 1

    def update_rule_collection_result_by_app_id_and_collection_id(self, app_id, rule_collection_id,
                                                                  rule_collection_version, result,
                                                                  first_hit_rule_code):
        if first_hit_rule_code is None:
            first_hit_rule_code = ""
        sql = "update app_rule_collection_results set " \
              "rule_collection_version = '{}', " \
              "result = {}, " \
              "first_hit_rule_code = '{}' " \
              "where app_id = '{}' and rule_collection_id = '{}'".format(rule_collection_version, result,
                                                                         first_hit_rule_code, app_id,
                                                                         rule_collection_id)
        return self.client.update(sql) == 1

    def upsert_rule_collection_result(self, app_id, rule_collection_id, rule_collection_version, result,
                                      first_hit_rule_code):
        rule_collection_record = self.get_rule_collection_result_by_app_id_and_collection_id(app_id, rule_collection_id)
        if rule_collection_record:
            return self.update_rule_collection_result_by_app_id_and_collection_id(app_id, rule_collection_id,
                                                                                  rule_collection_version, result,
                                                                                  first_hit_rule_code)
        else:
            return self.save_rule_collection_result(app_id, rule_collection_id, rule_collection_version, result,
                                                    first_hit_rule_code)

    def query_black_status(self, app_id):
        sql = "select count(1) cnt from app_rule_results where app_id='{}' and rule_type = 'A' and result=1;".format(app_id)
        query_rlt = self.client.query(sql)
        if query_rlt:
            return query_rlt[0]['cnt'] > 0
        else:
            return False
