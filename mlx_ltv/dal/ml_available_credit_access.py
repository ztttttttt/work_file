from mlx_database.dbdata import DBData


class MLAvailableCreditAccessData(DBData):
    def __init__(self, client=None, table=None):
        super(MLAvailableCreditAccessData, self).__init__(client, table)

    def save_credit(self, user_id, app_id, category_id, max_available_credit):
        sql = "insert into {} (user_id,app_id,category_id, max_available_credit) " \
              "values ('{}', '{}', '{}', {:.6f})".format(self.table, user_id, app_id,
                                                         category_id, max_available_credit)
        return self.client.update(sql) == 1
