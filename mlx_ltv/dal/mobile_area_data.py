from mlx_database.dbdata import DBData


class MobileAreaData(DBData):
    def __init__(self, client, table=None):
        super().__init__(client, table)

    def get_mobile_area(self, mobile):
        sql = "select province, city, service_provider from mobile_phone_area " \
              "where prefix = '{}'".format(mobile[:7])
        result = self.client.query(sql)
        if not result:
            return None, None, None
        return result[0]['province'], result[0]['city'], result[0]['service_provider']
