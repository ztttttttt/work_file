from mlx_database.dbdata import DBData


class JudgeData(DBData):
    def __init__(self, client, table=None):
        super().__init__(client, table)

    def get_judge_result_by_app_id_and_judge_id(self, app_id, judge_id):
        sql = "select judge_version, result, extension from app_judge_results " \
              "where app_id = '{}' and judge_id = '{}'".format(app_id, judge_id)
        return self.client.query(sql)

    def save_judge_result(self, app_id, judge_id, judge_version, result, extension):
        sql = "insert into app_judge_results (app_id, judge_id, judge_version, result, extension) " \
              "values ('{}', '{}', '{}', '{}', '{}')".format(app_id, judge_id, judge_version, result, extension)
        return self.client.update(sql) == 1

    def update_judge_result_by_app_id_and_judge_id(self, app_id, judge_id, judge_version, result, extension):
        sql = "update app_judge_results set " \
              "judge_version = '{}', " \
              "result = '{}', " \
              "extension = '{}' " \
              "where app_id = '{}' and judge_id = '{}'".format(judge_version, result, extension, app_id, judge_id)
        return self.client.update(sql) == 1

    def upsert_judge_result(self, app_id, judge_id, judge_version, result, extension):
        judge_result_record = self.get_judge_result_by_app_id_and_judge_id(app_id, judge_id)
        if judge_result_record:
            return self.update_judge_result_by_app_id_and_judge_id(app_id, judge_id, judge_version, result, extension)
        else:
            return self.save_judge_result(app_id, judge_id, judge_version, result, extension)
