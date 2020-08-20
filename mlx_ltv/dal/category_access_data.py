import json
from mlx_database.dbdata import DBData


class CategoryIDAccessData(DBData):
    def __init__(self, client=None, table=None):
        super(CategoryIDAccessData, self).__init__(client, table)

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


class ThresholdAccessData(DBData):
    MINIMUM_UPDATE_INTERVAL = 1  # minute

    def __init__(self, client=None, table=None):
        super(ThresholdAccessData, self).__init__(client, table)

    def read_threshold(self, category_id, model_info):
        sql = "select threshold from {}  " \
              "where category_id = '{}' and  model_info= '{}'" \
              "order by update_time desc".format(self.table, category_id, model_info)
        result = self.client.query_one(sql)
        return result if result else None

    def update_threshold(self, category_id, model_info, new_threshold, update_time):
        """
        check last category_threshold update time to prevent updating category_threshold twice within 1 minute,
        for distributed services
        """
        sql = "update {} set threshold = {:.6f}, update_time='{}' " \
              "where category_id = '{}' and model_info = '{}' " \
              "      and timestampdiff(minute, update_time, now()) >= {}" \
            .format(self.table, new_threshold, update_time, category_id, model_info, self.MINIMUM_UPDATE_INTERVAL)

        return self.client.update(sql) == 1


class ThresholdParameterAccessData(DBData):
    def __init__(self, client=None, table=None):
        super(ThresholdParameterAccessData, self).__init__(client, table)

    def read_parameter(self, category_id):
        sql = "select * from {} " \
              "where category_id = '{}'".format(self.table, category_id)
        category_param = self.client.query_one(sql)
        if not category_param:
            return None

        #convert json obj to python list
        category_param['pass_rate_bounds'] = json.loads(category_param['pass_rate_bounds'])
        category_param['threshold_delta_steps'] = json.loads(category_param['threshold_delta_steps'])
        return category_param


class StatisticCounterAccessData(DBData):
    def __init__(self, client=None, table=None):
        super(StatisticCounterAccessData, self).__init__(client, table)

    def read_statistic_counter(self, category_id):
        sql = "select app_counter,statistic_window,pass_rate_sample_size from {}  " \
              "where category_id = '{}'" \
              "order by update_time desc".format(self.table, category_id)
        result = self.client.query_one(sql)
        return result if result else None

    def update_counter(self, category_id):
        sql = "update {} set app_counter = app_counter + 1 " \
              "where category_id = '{}'".format(self.table, category_id)
        return self.client.update(sql) == 1


class ApplicationBriefAccessData(DBData):
    def __init__(self, client=None, table=None):
        super(ApplicationBriefAccessData, self).__init__(client, table)

    def create_app_record(self, app_id, category_id):
        sql = "insert into {} (app_id,category_id) " \
              "values ('{}', '{}')".format(self.table, app_id, category_id)
        return self.client.update(sql) == 1

    def read_record(self, app_id):
        sql = "select * from {} " \
              "where app_id = '{}' " \
              "order by create_time desc " \
              "limit 1".format(self.table, app_id)
        result = self.client.query_one(sql)
        return result if result else None

    def read_record_by_batch_size(self, category_id, batch_size):
        sql = "select is_pass,model_info from {} " \
              "where category_id = '{}'" \
              "order by create_time desc " \
              "limit {}".format(self.table, category_id, batch_size)

        return self.client.query(sql)

    def read_record_by_batch_period(self, category_id, model_info, start_time, end_time):
        sql = "select result, category_threshold, create_time from {} " \
              "where category_id = '{}' and model_info = '{}' and create_time > '{}' and create_time < '{}'" \
              "order by create_time desc ".format(self.table, category_id, model_info, start_time, end_time)

        return self.client.query(sql)

    def update_record(self, app_id, model_info, is_pass, check_type):
        sql = "update {} set model_info = '{}' ,is_pass={}, check_type='{}' " \
              "where app_id = '{}'".format(self.table, model_info, is_pass, check_type, app_id)
        return self.client.update(sql) == 1
