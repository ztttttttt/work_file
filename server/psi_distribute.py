import logging
import pickle
import pandas as pd
import math
import yaml
import os
import paramiko
import numpy as np
from database.mysql import MySql
from database.mongo import Mongo
from datetime import datetime, timedelta
from handle_exception.dingding_exception import DingdingExceptionHandler
from utillity.enum_processing import EnumMapper


class BaseFeatureMonitor(object):
    def __init__(self, config_file=None, product=None):
        self.config = self.set_config(config_file)
        self.mysql_risk = MySql(**self.config['mysql_risk'])
        self.mysql_risk_table = None
        self.mongo_derivable = Mongo(**self.config['mongo_derivable'])
        self.mongo_derivable_table = None
        self.except_handler = DingdingExceptionHandler(self.config['robots_psi'])
        self.product = product
        self.ssh_config = self.config['model_file_remote_ssh']

    def set_config(self, config_file):
        with open(config_file, 'r') as f:
            file = f.read()
        config = yaml.load(file)
        return config

    def get_model_path_from_mysql(self, table=None):
        pass

    def get_top_features(self):
        """"

        """
        model_path_list = self.get_model_path_from_mysql()
        model_path_df = pd.DataFrame(model_path_list)
        group_df = model_path_df.groupby('monitor_flag').apply(lambda x: x.model_path.unique()).rename('model_path_list').reset_index()
        group_df['top_features'] = group_df['model_path_list'].map(lambda x: self.top30_features(x))

        return group_df[['monitor_flag', 'top_features']]

    def top30_features(self, model_path):
        final_features = []
        # 连接远程服务器
        ssh_client = paramiko.Transport(self.ssh_config['hostname'], self.ssh_config['port'])
        ssh_client.connect(username=self.ssh_config['username'], password=self.ssh_config['password'])
        sftp = paramiko.SFTPClient.from_transport(ssh_client)

        for remote_model_path in model_path:
            # remote_model_path = model_dict['model_path']
            # 判断本地模型文件所在目录是否存在，没有就创建
            if not os.path.isdir(os.path.split(remote_model_path)[0]):
                os.makedirs(os.path.split(remote_model_path)[0])
                # 将远程文件下载到本地
                sftp.get(remote_model_path, remote_model_path)

            with open(remote_model_path, 'rb') as f:
                model_info = pickle.load(f)
            top_columns = []
            try:
                model = model_info['model']
                enum = model.get_params()['enum']
                mm = model.get_params()['clf']
                top_columns = []
                for i, v in enumerate(
                        sorted(zip(map(lambda x: round(x, 4), mm.feature_importances_), enum.clean_col_names),
                               reverse=True)):
                    if i <= 30:
                        top_columns.append(v[1])
            except Exception as e:
                logging.error(e)
            final_features.extend(top_columns)
        sftp.close()
        no_final_features = ['ALPHA_Behavior_submit_date', 'ALPHA_Behavior_submit_hour',
                             'ALPHA_Behavior_submit_weekday', 'X_DNA_Behavior_submit_date',
                             'X_DNA_Behavior_submit_hour', 'X_DNA_Behavior_submit_weekday']  # 这些不监控
        final_features = list(set(final_features) - set(no_final_features))
        # logging.info('{}-top_features: {}'.format(self.product, final_features))
        return final_features

    def get_appid_from_mysql(self, start_time, diff_day, diff_hour):
        """获取所需要的11天的所有appid信息"""
        end_time = (datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S') + timedelta(days=diff_day)).strftime(
            "%Y-%m-%d %H:%M:%S")
        start_hour = 0
        end_hour = start_hour + diff_hour
        sql = '''select upper(app_id) as app_id,flow_type,work_flag,date(create_time) as date
                        from {}
                        where create_time >= '{}'
                              and create_time < '{}' 
                              and hour(create_time) >= {}
                              and hour(create_time) <= {}
             '''.format(self.mysql_risk_table, start_time, end_time, start_hour, end_hour)
        res = self.mysql_risk.query(sql)
        return pd.DataFrame(res)

    def get_features(self, df_appid, top_feature):
        appids = list(set(df_appid['app_id'].tolist()))
        qry = {'_id': {'$in': appids}}
        qry1 = {feature: 1 for feature in top_feature}
        res = self.mongo_derivable.get_collection(self.mongo_derivable_table).find(qry, qry1, batch_size=500)
        res_list = list(res)
        return pd.DataFrame(res_list)

    @staticmethod
    def cal_psi(x, y):
        e = 0.00001
        if x == 0 or y == 0:
            psi = (x - y) * math.log((x + e) / (y + e))
        else:
            psi = (x - y) * math.log(x/y)
        return round(psi, 3)

    def psi(self, df_feature_1, df_feature_2, feature, bin_num=10):
        df_feature_1['label'] = 0
        df_feature_2['label'] = 1
        df_feature = pd.concat([df_feature_1, df_feature_2])
        df_feature = df_feature.replace('null', np.nan)
        df_feature = df_feature.replace('NaN', np.nan)
        df_feature = df_feature.apply(pd.to_numeric, errors='ignore')
        enum = EnumMapper(maximum_enum_num=100)
        enum.fit(df_feature)
        df_feature = enum.transform(df_feature)
        if feature in df_feature.columns.tolist():
            df_psi = df_feature[[feature, 'label']].copy()
            if df_psi[feature].dtype not in ['int', 'float'] and df_psi[feature].unique().shape[0] > 20:
                # print("The unique number of feature is {}".format(df_psi[feature].unique().shape[0]))
                return None, 999
            else:
                if df_psi[feature].unique().shape[0] > 2:
                    df_psi['bins'] = pd.qcut(df_psi[feature], 10, precision=2, duplicates='drop')
                    nan_df = df_psi[df_psi[feature].map(lambda x: pd.isnull(x))].reset_index(drop=True)
                    if not nan_df.empty:
                        df_psi['bins'] = df_psi['bins'].cat.add_categories('(-999.1, -999]')
                        df_psi['bins'] = df_psi['bins'].fillna('(-999.1, -999]')
                else:
                    df_psi['bins'] = df_psi[feature].map(lambda x: -999 if pd.isnull(x) else x)
                group_df = df_psi.groupby(['bins', 'label']).size().unstack('label')
                group_df = group_df.fillna(0)
                group_df['b_rate'] = group_df[0] / group_df[0].sum()
                group_df['a_rate'] = group_df[1] / group_df[1].sum()
                group_df = group_df.map(lambda x: round(x, 4))
                group_df['psi_part'] = list(map(lambda x, y: self.cal_psi(x, y), group_df.b_rate, group_df.a_rate))
                group_df = group_df.apply(lambda x: round(x, 3))
                group_df = group_df.reset_index()

                return group_df, group_df.psi_part.sum()
        else:
            return None, 99

    @staticmethod
    def define_mf(x, y):
        if x == 'c' and y == 'precheck':
            return 'cp'
        elif x == 'c' and y == 'finalcheck':
            return 'cf'
        elif x == 'f' and y == 'precheck':
            return 'fp'
        elif x == 'f' and y == 'finalcheck':
            return 'ff'
        elif x == 'w' and y == 'precheck':
            return 'wp'
        elif x == 'w' and y == 'finalcheck':
            return 'wf'
        elif x == 'q' and y == 'finalcheck':
            return 'q'

    def psi_classified(self, start_time, diff_day, diff_hour, timedetail):
        """psi分类监控"""
        total_appids_df = self.get_appid_from_mysql(start_time, diff_day, diff_hour)  # 获取所需要的11天的所有appid信息
        total_appids_df.date = total_appids_df.date.map(lambda x: str(x))  # 将里面date字段的类型转换为str
        total_appids_df['monitor_flag'] = list(map(lambda x, y: self.define_mf(x, y),
                                                   total_appids_df.flow_type,
                                                   total_appids_df.work_flag))
        cp_ls_top_psi = []  # 发送到钉钉的psi监控列表
        features_df = self.get_top_features()
        for monitor_flag in total_appids_df.monitor_flag.unique().tolist():
            top_features = features_df.query("monitor_flag=='{}'".format(monitor_flag)).top_features.values
            if top_features:
                top_psi = self.psi_distr(start_time, total_appids_df, top_features[0], monitor_flag)
                if monitor_flag == 'cp' and top_psi:
                    cp_ls_top_psi.append('=======开卡初审=======')
                elif monitor_flag == 'cf' and top_psi:
                    cp_ls_top_psi.append('=======开卡复审=======')
                elif monitor_flag == 'fp' and top_psi:
                    cp_ls_top_psi.append('=======首贷提现初审=======')
                elif monitor_flag == 'ff' and top_psi:
                    cp_ls_top_psi.append('=======首贷提现复审=======')
                elif monitor_flag == 'wp' and top_psi:
                    cp_ls_top_psi.append('=======复贷初审=======')
                elif monitor_flag == 'wf' and top_psi:
                    cp_ls_top_psi.append('=======复贷复审=======')
                elif monitor_flag == 'q' and top_psi:
                    cp_ls_top_psi.append('=======结清调额=======')
                else:
                    pass
                cp_ls_top_psi.extend(top_psi)
        logging.info('warming psi list: {}'.format(cp_ls_top_psi))
        if cp_ls_top_psi:
            cp_ls_top_psi.insert(0, '*******{}-psi报警*******'.format(self.product))
            cp_ls_top_psi.insert(1, '时间：{}'.format(datetime.now().strftime('%Y-%m-%d ') + timedetail))
            self.except_handler.handle(msg=cp_ls_top_psi)

    def psi_distr(self, start_time, total_appids_df, top_features, monitor_flag):
        the_psi_date = (datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S') + timedelta(days=10)).strftime(
            '%Y-%m-%d')  # 所监控的日期
        logging.info('所监控的日期为：{}'.format(the_psi_date))
        # 所监控日期前十天对应类型的appids
        df_appid1 = total_appids_df.query(
            "monitor_flag=='{}' and date!='{}'".format(monitor_flag, the_psi_date)).reset_index(drop=True)
        df_appid1 = df_appid1.sample(min(10000, df_appid1.shape[0]))
        logging.info('monitor_flag:{} 前十天全部app_id个数：{}'.format(monitor_flag, len(df_appid1)))

        # 所监控日期对应类型的appids
        df_appid2 = total_appids_df.query(
            "monitor_flag=='{}' and date=='{}'".format(monitor_flag, the_psi_date)).reset_index(drop=True)
        df_appid2 = df_appid2.sample(min(1000, df_appid2.shape[0]))
        logging.info('monitor_flag:{}  所监控的app_id个数：{}'.format(monitor_flag, len(df_appid2)))
        ls_top_psi = []
        df_feature_all_1 = self.get_features(df_appid1, top_features)
        df_feature_all_2 = self.get_features(df_appid2, top_features)
        psi_dict = {}
        for feature in top_features:
            df_feature_1 = pd.DataFrame(df_feature_all_1, columns=[feature])
            df_feature_2 = pd.DataFrame(df_feature_all_2, columns=[feature])
            df, psi = self.psi(df_feature_1, df_feature_2, feature, bin_num=10)
            psi_dict.update({feature: psi})
            if psi > 0.25:
                ls_top_psi.append("{}--psi:{}".format(feature, round(psi, 3)))
                df['bins'] = df['bins'].map(lambda x: str(x))
                max_index = df.query("psi_part=={}".format(df.psi_part.max()))['bins'].values[0]
                str_text = ''
                if str(max_index) == '(-999.1, -999]':
                    str_text += '原因：缺失值变化导致, '
                else:
                    str_text += '原因：区间{}变化所致, '.format(max_index)
                if df.query("bins=='{}'".format(max_index)).a_rate.values[0] > \
                        df.query("bins=='{}'".format(max_index)).b_rate.values[0]:
                    str_text += '当前比例大于过去比例。'
                else:
                    str_text += '当前比例小于过去比例。'
                ls_top_psi.append(str_text)
                ls_top_psi.append('==' * 18)
                ls_top_psi.append(str(df))
                ls_top_psi.append('==' * 18)
        logging.info('{} calculate psi done :{}'.format(monitor_flag, psi_dict))
        return ls_top_psi

    # 前一天
    def job1(self):
        """比较昨天top特征的分布与昨天的前10天top特征的分布"""
        logging.info('{} start handle psi_monitor job1!'.format(self.product))
        start_time = (datetime.now() - timedelta(days=11)).strftime(
            '%Y-%m-%d') + ' 00:00:00'  # 获取所监控及其对比的前10天所有appid的开始时间
        diff_day = 11  # 获取从开始时间往后11天的数据
        diff_hour = 24  # 获取每天从0时到24时的数据
        self.psi_classified(start_time, diff_day, diff_hour, timedetail='前一天0-24时分布变化')
        logging.info('{} end handle psi_monitor job1!'.format(self.product))

    # 当天
    def job2(self):
        """比较当天(0-16时)的top特征的分布与前10天(0-16时)top特征的分布"""
        logging.info('{} start handle feature_monitor job2!'.format(self.product))
        start_time = (datetime.now() - timedelta(days=10)).strftime(
            '%Y-%m-%d') + ' 00:00:00'  # 获取所监控及其对比的前10天所有appid的开始时间
        diff_day = 11  # 获取从开始时间往后11天的数据
        diff_hour = 15  # 获取每天从0时到15时的数据
        self.psi_classified(start_time, diff_day, diff_hour, timedetail='当天0-16时分布变化')
        logging.info('{} end handle psi_monitor job2!'.format(self.product))

    # 当天
    def job3(self):
        """比较当天(0-18时)的top特征的分布与前10天(0-18时)top特征的分布"""
        logging.info('{} start handle feature_monitor job2!'.format(self.product))
        start_time = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d') + ' 00:00:00'
        diff_day = 11  # 获取从开始时间往后11天的数据
        diff_hour = 17  # 获取每天从0时到18时的数据
        self.psi_classified(start_time, diff_day, diff_hour, timedetail='当天0-18时分布变化')
        logging.info('{} end handle psi_monitor job3!'.format(self.product))


class AlphaPsiMonitor(BaseFeatureMonitor):
    def __init__(self, config_file='config/alphaConfig.yaml', product='alpha'):
        super(AlphaPsiMonitor, self).__init__(config_file, product)
        self.mysql_risk_table = 'alpha_category_results'
        self.mongo_derivable_table = 'derivativevariables'

    def get_model_path_from_mysql(self, table='alpha_model_configs'):
        sql = 'SELECT DISTINCT model_path as model_path,monitor_flag FROM {} WHERE is_setup=1 AND id not in (1,51,101,201);'.format(
            table)
        model_path_list = self.mysql_risk.query(sql)
        return model_path_list


class JdxPsiMonitor(BaseFeatureMonitor):
    def __init__(self, config_file='config/jdxConfig.yaml', product='jdx'):
        super(JdxPsiMonitor, self).__init__(config_file, product)
        self.mysql_risk_table = 'jdx_category_results'
        self.mongo_derivable_table = 'derivables'

    def get_model_path_from_mysql(self, table='jdx_category_model_relations'):
        sql = u"SELECT DISTINCT estor_path as model_path,monitor_flag FROM {} WHERE is_setup=1 and threshold !=1 " \
              u"and id not in (1,101,201,202,251,261) and monitor_flag in ('wp', 'wf', 'q');".format(table)
        model_path_list = self.mysql_risk.query(sql)
        return model_path_list


class LtvPsiMonitor(BaseFeatureMonitor):
    def __init__(self, config_file='config/ltvConfig.yaml', product='ltv'):
        super(LtvPsiMonitor, self).__init__(config_file, product)
        self.mysql_risk_table = 'machine_learning_risk_model_results'
        self.mongo_derivable_table = 'derivativevariables'

    def get_model_path_from_mysql(self, table='ltv_ml_model_configs'):
        sql = "SELECT DISTINCT model_path as model_path FROM {} WHERE is_setup=1 and category_id in " \
              "('a5c22eb8-f1fd-11e7-9f36-6c4008b8a73e','a5c22eb8-f1fd-hb01-9f36-6c4008b8a73e');".format(table)
        model_path_list = self.mysql_risk.query(sql)
        return model_path_list

    def get_top_features(self):
        model_path_list = self.get_model_path_from_mysql()
        final_features = []
        # 连接远程服务器
        ssh_client = paramiko.Transport(self.ssh_config['hostname'], self.ssh_config['port'])
        ssh_client.connect(username=self.ssh_config['username'], password=self.ssh_config['password'])
        sftp = paramiko.SFTPClient.from_transport(ssh_client)
        for model_dict in model_path_list:
            remote_model_path = model_dict['model_path']
            # 判断本地模型文件所在目录是否存在，没有就创建
            if not os.path.isdir(os.path.split(remote_model_path)[0]):
                os.makedirs(os.path.split(remote_model_path)[0])
                # 将远程文件下载到本地
                sftp.get(remote_model_path, remote_model_path)
            with open(remote_model_path, 'rb') as f:
                model_info = pickle.load(f)
            top_columns = []
            try:
                model = model_info['model']
                enum = model.get_params()['enum']
                mm = model.get_params()['clf']
                top_columns = []
                for i, v in enumerate(
                        sorted(zip(map(lambda x: round(x, 4), mm.feature_importances_), enum.clean_col_names),
                               reverse=True)):
                    if i <= 30:
                        top_columns.append(v[1])
            except Exception as e:
                logging.error(e)
            final_features.extend(top_columns)
        sftp.close()
        final_features = list(set(final_features))
        logging.info('{}-top_features: {}'.format(self.product, final_features))
        return final_features

    # 获取某个时间段的经过审核的appid
    def appid_confirm(self, start_time_str, diff_hour, start_hour, end_hour):
        end_time = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S') + timedelta(hours=diff_hour)
        end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
        sql = '''select upper(app_id) as app_id
                        from {}
                        where create_time >= '{}'
                              and create_time <= '{}' 
                              and hour(create_time) >= {}
                              and hour(create_time) <= {}
             '''.format(self.mysql_risk_table, start_time_str, end_time_str, start_hour, end_hour)
        res = self.mysql_risk.query(sql)
        return pd.DataFrame(res)

    def psi_distr(self, start_time_str, top_features, diff_hour, timedetail):
        start_time_before = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S') - timedelta(days=10)
        start_time_str_before = start_time_before.strftime("%Y-%m-%d %H:%M:%S")
        start_hour = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S').hour
        end_hour = start_hour + diff_hour  # 24
        df_appid1 = self.appid_confirm(start_time_str_before, diff_hour=240, start_hour=start_hour, end_hour=end_hour)
        df_appid1 = df_appid1.sample(min(10000, df_appid1.shape[0]))
        logging.info('前十天全部app_id个数：{}'.format(len(df_appid1)))
        df_appid2 = self.appid_confirm(start_time_str, diff_hour=diff_hour, start_hour=start_hour, end_hour=end_hour)
        df_appid2 = df_appid2.sample(min(1000, df_appid2.shape[0]))
        logging.info('所监控的app_id个数：{}'.format(len(df_appid2)))
        ls_top_psi = []
        psi_dict = {}
        df_feature_all_1 = self.get_features(df_appid1, top_features)
        df_feature_all_2 = self.get_features(df_appid2, top_features)
        for feature in top_features:
            df_feature_1 = pd.DataFrame(df_feature_all_1, columns=[feature])
            df_feature_2 = pd.DataFrame(df_feature_all_2, columns=[feature])
            # psi监控
            df, psi = self.psi(df_feature_1, df_feature_2, feature, bin_num=10)
            psi_dict.update({feature: psi})
            if psi > 0.25:
                ls_top_psi.append("{}--psi:{}".format(feature, round(psi, 3)))
                df['bins'] = df['bins'].map(lambda x: str(x))
                max_index = df.query("psi_part=={}".format(df.psi_part.max()))['bins'].values[0]
                str_text = ''
                if str(max_index) == '(-999.1, -999]':
                    str_text += '原因：缺失值变化导致, '
                else:
                    str_text += '原因：区间{}变化所致, '.format(max_index)
                if df.query("bins=='{}'".format(max_index)).a_rate.values[0] > \
                        df.query("bins=='{}'".format(max_index)).b_rate.values[0]:
                    str_text += '当前比例大于过去比例。'
                else:
                    str_text += '当前比例小于过去比例。'
                ls_top_psi.append(str_text)
                ls_top_psi.append('==' * 18)
                ls_top_psi.append(str(df))
                ls_top_psi.append('==' * 18)
        logging.info('ltv calculate psi done :{}'.format(psi_dict))
        if ls_top_psi:
            ls_top_psi.insert(0, '*******{}-psi报警*******'.format(self.product))
            ls_top_psi.insert(1, '时间：{}'.format(datetime.now().strftime('%Y-%m-%d ') + timedetail))
            self.except_handler.handle(msg=ls_top_psi)

    # 前一天
    def job1(self):
        # self.except_handler.handle(msg="当天8点执行，比较前一天LTV top特征的分布与前30天top特征的分布")
        logging.info('{} start handle psi_monitor job1!'.format(self.product))
        start_time_str = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d') + ' 00:00:00'
        top_features = self.get_top_features()
        self.psi_distr(start_time_str, top_features, diff_hour=24, timedetail='前一天0-24时分布变化')
        logging.info('{} end handle psi_monitor job1!'.format(self.product))

    # 当天
    def job2(self):
        logging.info('{} start handle psi_monitor job2!'.format(self.product))
        start_time_str = (datetime.now() - timedelta(days=0)).strftime('%Y-%m-%d') + ' 00:00:00'
        top_features = self.get_top_features()
        self.psi_distr(start_time_str, top_features, diff_hour=16, timedetail='当天0-16时分布变化')
        logging.info('{} end handle psi_monitor job2!'.format(self.product))

    # 当天
    def job3(self):
        logging.info('{} start handle psi_monitor job3!'.format(self.product))
        start_time_str = (datetime.now() - timedelta(days=0)).strftime('%Y-%m-%d') + ' 00:00:00'
        top_features = self.get_top_features()
        self.psi_distr(start_time_str, top_features, diff_hour=18, timedetail='当天0-18时分布变化')
        logging.info('{} end handle psi_monitor job3!'.format(self.product))


if __name__ == "__main__":
    # start_time_str = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d') + ' 00:00:00'
    # print(start_time_str)
    # LtvPsiMonitor = LtvPsiMonitor()
    # AlphaPsiMonitor = AlphaPsiMonitor()
    # LtvPsiMonitor.job1()
    # AlphaPsiMonitor.job1()
    import sys
    # sys.path.insert(0, os.path.abspath("../mlDataMonitor"))
    # config_file = 'config/alphaConfig.yaml'
    # def set_config(config_file):
    #     with open(config_file, 'r') as f:
    #         file = f.read()
    #     config = yaml.load(file)
    #     return config
    #
    #
    # def psi(df_feature_1, df_feature_2, feature, bin_num=10):
    #     df_feature_1['label'] = 0
    #     df_feature_2['label'] = 1
    #     df_feature = pd.concat([df_feature_1, df_feature_2])
    #     df_feature = df_feature.replace('null', np.nan)
    #     df_feature = df_feature.replace('NaN', np.nan)
    #     # df_feature = df_feature.apply(pd.to_numeric, errors='ignore')
    #     # enum = EnumMapper(maximum_enum_num=100)
    #     # enum.fit(df_feature)
    #     # df_feature = enum.transform(df_feature)
    #     if feature in df_feature.columns.tolist():
    #         df_psi = df_feature[[feature, 'label']].copy()
    #         if df_psi[feature].dtype not in ['int', 'float', 'int64', 'float64'] and df_psi[feature].unique().shape[0] > 20:
    #             # print("The unique number of feature is {}".format(df_psi[feature].unique().shape[0]))
    #             return None, 999
    #         else:
    #             if df_psi[feature].unique().shape[0] > 2:
    #                 df_psi['bins'] = pd.qcut(df_psi[feature], 10, precision=2, duplicates='drop')
    #                 nan_df = df_psi[df_psi[feature].map(lambda x: pd.isnull(x))].reset_index(drop=True)
    #                 if not nan_df.empty:
    #                     df_psi['bins'] = df_psi['bins'].cat.add_categories('(-999.1, -999]')
    #                     df_psi['bins'] = df_psi['bins'].fillna('(-999.1, -999]')
    #             else:
    #                 df_psi['bins'] = df_psi[feature].map(lambda x: -999 if pd.isnull(x) else x)
    #             group_df = df_psi.groupby(['bins', 'label']).size().unstack('label')
    #             group_df = group_df.fillna(0)
    #             group_df['b_rate'] = group_df[0] / group_df[0].sum()
    #             group_df['a_rate'] = group_df[1] / group_df[1].sum()
    #             e = 0.000000000001
    #             group_df['psi_part'] = group_df.apply(
    #                 lambda group_df: (group_df['a_rate'] - group_df['b_rate']) * math.log(
    #                     (group_df['a_rate'] + e) / (group_df['b_rate'] + e)), axis=1)
    #             group_df['psi_part'] = group_df['psi_part'].map(lambda x: round(x, 3))
    #             group_df = group_df.apply(lambda x: round(x, 3))
    #             # group_df.index.name = ''
    #
    #             return group_df, group_df.psi_part.sum()
    #     else:
    #         return None, 99
    #
    # config = set_config("C:/Users/zhut\Desktop/projects_now/mlDataMonitor/config/alphaConfig.yaml")
    # except_handler = DingdingExceptionHandler(config['robots_psi'])
    # ls_top_psi = []
    # ls_top_psi_1 = []
    # df_feature_1 = pd.read_csv('C:/Users/zhut/Desktop/dfg1.csv')
    # df_feature_2 = pd.read_csv('C:/Users/zhut/Desktop/dfg2.csv')
    # # print(df_feature_1.flow_type.unique())
    # for feature in ['ALPHA_Behavior_all_orders_max_overdue_days ',
    #                 'ALPHA_XINYAN_RadarV2_apply_report_detail_A22160006 ']:
    #     df1 = df_feature_1.query("flow_type=='w'")
    #     df2 = df_feature_2.query("flow_type=='w'")
    #     df_feature1 = pd.DataFrame(df1, columns=[feature])
    #     df_feature2 = pd.DataFrame(df2, columns=[feature])
    #     # psi监控
    #     df, psi = psi(df_feature2, df_feature1, feature, bin_num=10)
    #     print(df)
    #     if psi > 0.25:
    #         ls_top_psi.append("{}--psi:{}".format(feature, round(psi, 3)))
    #         df = df.reset_index()
    #         df['bins'] = df['bins'].map(lambda x: str(x))
    #         max_index = df.query("psi_part=={}".format(df.psi_part.max()))['bins'].values[0]
    #         if str(max_index) == '(-999.1, -999]':
    #             str_text = '原因：缺失值变化导致, '
    #         else:
    #             str_text = '原因：区间{}的变化所致, '.format(max_index)
    #         if df.query("bins=='{}'".format(max_index)).a_rate.values[0] > df.query("bins=='{}'".format(max_index)).b_rate.values[0]:
    #             str_text += '当前比例大于过去比例。'
    #         else:
    #             str_text += '当前比例小于过去比例。'
    #         ls_top_psi.append(str_text)
    #         ls_top_psi.append('==' * 18)
    #         ls_top_psi.append(str(df))
    #         ls_top_psi.append('==' * 18)
    # print(ls_top_psi)

    # if ls_top_psi:
    #     ls_top_psi.insert(0, '*******{}-psi报警*******'.format('ltv'))
    #     ls_top_psi.insert(1, '时间：{}'.format(datetime.now().strftime('%Y-%m-%d ') + '当天0-16时'))
    # for i in ls_top_psi:
    #     print(i)
        # except_handler.handle(msg=ls_top_psi)
