from mlx_database.dbdata import DBData


class MLResultAccessData(DBData):
    def __init__(self, client=None, table=None):
        super(MLResultAccessData, self).__init__(client, table)

    def save_result(self, user_id, app_id, category_id, result, threshold, model_info):
        sql = "insert into {} (user_id,app_id,category_id, result, threshold, model_info) " \
              "values ('{}', '{}', '{}', {:.6f}, {:.6f}, '{}')".format(self.table, user_id, app_id,
                                                               category_id, result, threshold, model_info)
        return self.client.update(sql) == 1

