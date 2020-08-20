import logging

import schedule
import time
import pandas as pd
import yaml
from datetime import datetime, timedelta

from database.mysql import MySql
from handle_exception.dingding_exception import DingdingExceptionHandler


class BasePassMonitor(object):
    def __init__(self, config_file=None, product=None):
        self.config = self.set_config(config_file)
        self.mysql_risk = MySql(**self.config['mysql_risk'])
        self.except_handler = DingdingExceptionHandler(self.config['robots'])
        self.product = product

    def set_config(self, config_file):
        with open(config_file, 'r') as f:
            file = f.read()
        config = yaml.load(file)
        return config

    def pass_ratio(self):
       pass


class LtvPassMonitor(BasePassMonitor):
    def __init__(self, config_file='./config/ltvConfig.yaml', product='ltv'):
        super(LtvPassMonitor, self).__init__(config_file, product)

    def pass_ratio(self):
        sql = '''select * from 
                (
                select create_time_str,
                sum(case when category_id = 'a5c22eb8-f1fd-11e7-9f36-6c4008b8a73e' then 1 else 0 end) as reloan_num ,
                sum(case when final_result = 'pass' and category_id = 'a5c22eb8-f1fd-11e7-9f36-6c4008b8a73e' then 1 else 0 end) / 
                    sum(case when category_id = 'a5c22eb8-f1fd-11e7-9f36-6c4008b8a73e' then 1 else 0 end) as reloan_pass_rate
                from 
                (select t1.*,  date(t1.create_time) as create_time_str    ,t2.result as final_result ,t2.extension  as extension,
                                            t3.max_available_credit as max_available_credit
                                     from  machine_learning_risk_model_results t1
                                     inner join  app_judge_results as t2
                                     on t1.app_id = t2.app_id
                                     inner join machine_learning_max_available_credits as t3
                                     on t1.app_id = t3.app_id
                                     where t1.create_time > date_add(current_date(), interval -1 day)  
                )tt
                group by create_time_str
                )tt1 

                LEFT JOIN


                #new model pass rate of first loan
                (
                select create_time_str,
                sum(case when category_id = '34423181-e5fe-11e7-93e1-6c0b84a6f2b3' then 1 else 0 end) as first_loan_num ,
                sum(case when final_result = 'pass' and category_id = '34423181-e5fe-11e7-93e1-6c0b84a6f2b3' then 1 else 0 end) / 
                   sum(case when category_id = '34423181-e5fe-11e7-93e1-6c0b84a6f2b3' then 1 else 0 end) as first_loan_pass_rate
                from 
                (select t1.*,  date(t1.create_time) as create_time_str    ,t2.result as final_result ,t2.extension  as extension,
                                            t3.max_available_credit as max_available_credit
                                     from  machine_learning_risk_model_results t1
                                     inner join  app_judge_results as t2
                                     on t1.app_id = t2.app_id
                                     inner join machine_learning_max_available_credits as t3
                                     on t1.app_id = t3.app_id
                                     where t1.create_time > date_add(current_date(), interval -1 day)  
                )tt
                group by create_time_str
                )tt2
                on tt1.create_time_str = tt2.create_time_str 
              '''
        df_tmp = pd.DataFrame(self.mysql_risk.query(sql))
        df_tmp['create_time_str'] = df_tmp['create_time_str'].apply(lambda x: str(x))
        df_tmp = df_tmp.set_index(['create_time_str'])
        return df_tmp

    def job1(self):
        pass_rate_dict = self.pass_ratio()
        # 当前时间前一天所在天数通过率
        yestoday_str = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        first_loan_pass_rate = pass_rate_dict.loc[yestoday_str, 'first_loan_pass_rate']
        reloan_pass_rate = pass_rate_dict.loc[yestoday_str, 'reloan_pass_rate']
        if first_loan_pass_rate < self.config['first_loan_pass_rate_threshold']:
            self.except_handler.handle(msg='{}---first_loan_pass_rate:{}    first_loan_pass_rate_threshold:{}'.format(self.product, first_loan_pass_rate,self.config['first_loan_pass_rate_threshold']))
        if reloan_pass_rate < self.config['reloan_pass_rate_threshold']:
            self.except_handler.handle(msg='{}---reloan_pass_rate:{}    reloan_pass_rate_threshold:{}'.format(self.product, reloan_pass_rate,self.config['reloan_pass_rate_threshold']))

    def job2(self):
        pass_rate_dict = self.pass_ratio()
        # 当前时间所在天数通过率
        now_str = datetime.now().strftime('%Y-%m-%d')
        first_loan_pass_rate = pass_rate_dict.loc[now_str, 'first_loan_pass_rate']
        reloan_pass_rate = pass_rate_dict.loc[now_str, 'reloan_pass_rate']
        if first_loan_pass_rate < self.config['first_loan_pass_rate_threshold']:
            self.except_handler.handle(msg='{}---first_loan_pass_rate:{}    first_loan_pass_rate_threshold:{}'.format(self.product, first_loan_pass_rate,self.config['first_loan_pass_rate_threshold']))
        if reloan_pass_rate < self.config['reloan_pass_rate_threshold']:
            self.except_handler.handle(msg='{}---reloan_pass_rate:{}    reloan_pass_rate_threshold:{}'.format(self.product, reloan_pass_rate,self.config['reloan_pass_rate_threshold']))


class OmegaPassMonitor(BasePassMonitor):
    def __init__(self, config_file='./config/omegaConfig.yaml', product='omega'):
        super(OmegaPassMonitor, self).__init__(config_file, product)

    def pass_ratio(self):
        sql = '''select date(t1.create_time) as create_time_str , 
                sum(case when t2.is_pass = 1 and t1.result = 1 then 1 else 0 end) / count(distinct t1.app_id) as pass_rate
                from omega_ml_rule_check_result as t1 
                inner join omega_ml_model_results as t2
                   on t1.app_id = t2.app_id
                where t1.create_time >  date_add(current_date(), interval -1 day) 
                    and t1.channel != 13
                    and t1.service = 'omega_open_card'
                group by date(t1.create_time)'''
        df_tmp = pd.DataFrame(self.mysql_risk.query(sql))
        df_tmp['create_time_str'] = df_tmp['create_time_str'].apply(lambda x: str(x))
        df_tmp = df_tmp.set_index(['create_time_str'])
        return df_tmp

    def job1(self):
        pass_rate_dict = self.pass_ratio()
        # 当前时间前一天所在天数通过率
        yestoday_str = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        pass_rate = pass_rate_dict.loc[yestoday_str, 'pass_rate']
        if pass_rate < self.config['pass_rate_threshold']:
            self.except_handler.handle(msg='{}---pass_rate:{}    pass_rate_threshold:{}'.format(self.product, pass_rate,self.config['first_loan_pass_rate_threshold']))


    def job2(self):
        pass_rate_dict = self.pass_ratio()
        # 当前时间所在天数通过率
        now_str = datetime.now().strftime('%Y-%m-%d')
        pass_rate = pass_rate_dict.loc[now_str, 'pass_rate']
        if pass_rate < self.config['pass_rate_threshold']:
            self.except_handler.handle(msg='{}---pass_rate:{}    pass_rate_threshold:{}'.format(self.product, pass_rate,self.config['pass_rate_threshold']))


class JdxPassMonitor(BasePassMonitor):
    def __init__(self, config_file='./config/jdxConfig.yaml', product='jdx'):
        super(JdxPassMonitor, self).__init__(config_file, product)

    def pass_ratio(self):
        sql = '''select tt1.create_time_str, tt1.opencard_pass_rate, tt2.reloan_pass_rate, tt3.diversion_pass_rate from 
                (
                SELECT  DATE(b.create_time) as create_time_str,sum(if(a.result>0 and b.credit_ml>0 and b.judge_by=1, 1, 0))/sum(if( b.judge_by=1, 1, 0)) as opencard_pass_rate
                FROM prod_ml_jdcl.jd_ml_rule_check_result a, prod_ml_jdcl.machine_learning_jdcl_results b
                WHERE a.app_id = b.app_id
                  and b.create_time>date_add(current_date(), interval -1 day)
                  AND b.category_id in ('jdownf8a-42d7-11e8-a406-6c4008b8a73e',
                                        'def79688-42d7-11e8-ad52-6c4008b8a73e',
                                        'jdblkce2-d5c5-11e8-81a5-6c4008b8a73e',
                                        'jdjieju-d5c5-11e8-81a5-6c4008b8a73e')
                GROUP BY DATE(b.create_time)
                ) tt1
                
                LEFT JOIN
                
                (
                select dis.dt as create_time_str,
                          sum(if(dis.result>=dis.threshold and dis.rule_result=1,1,0))/count(dis.user_id) as reloan_pass_rate 
                from
                 (select distinct rule_res.user_id,ml_res.result,ml_res.threshold, ml_res.model,
                               rule_res.result as rule_result,DATE(ml_res.create_time) as dt
                    from jd_ml_rule_check_result as rule_res
                    join jdx_withdraw_model_results as ml_res
                       on rule_res.app_id=ml_res.app_id
                    where ml_res.category_id='reln6662-c47c-11e8-9b91-6c4008b8a73e'
                        and rule_res.rule_type='w'
                      and ml_res.create_time>date_add(current_date(), interval -1 day))  as dis
                group by dis.dt
                ) tt2
                on tt1.create_time_str = tt2.create_time_str
                
                LEFT JOIN
                
                (
                SELECT  DATE(b.create_time) as create_time_str,sum(if(a.result>0 and b.credit_ml>0 and b.judge_by=2, 1, 0))/sum(if( b.judge_by=2, 1, 0)) as diversion_pass_rate
                FROM prod_ml_jdcl.jd_ml_rule_check_result a, prod_ml_jdcl.machine_learning_jdcl_results b
                WHERE a.app_id = b.app_id
                  and b.create_time>date_add(current_date(), interval -1 day)
                  AND b.category_id in ('jdownf8a-42d7-11e8-a406-6c4008b8a73e',
                                        'def79688-42d7-11e8-ad52-6c4008b8a73e',
                                        'jdblkce2-d5c5-11e8-81a5-6c4008b8a73e',
                                        'jdjieju-d5c5-11e8-81a5-6c4008b8a73e')
                GROUP BY DATE(b.create_time)
                ) tt3
                
                on tt1.create_time_str = tt3.create_time_str'''
        df_tmp = pd.DataFrame(self.mysql_risk.query(sql))
        df_tmp['create_time_str'] = df_tmp['create_time_str'].apply(lambda x: str(x))
        df_tmp = df_tmp.set_index(['create_time_str'])
        return df_tmp

    def job1(self):
        pass_rate_dict = self.pass_ratio()
        # 当前时间前一天所在天数通过率
        yestoday_str = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        opencard_pass_rate = pass_rate_dict.loc[yestoday_str, 'opencard_pass_rate']
        reloan_pass_rate = pass_rate_dict.loc[yestoday_str, 'reloan_pass_rate']
        diversion_pass_rate = pass_rate_dict.loc[yestoday_str, 'diversion_pass_rate']
        if opencard_pass_rate < self.config['opencard_pass_rate_threshold']:
            self.except_handler.handle(
                msg='{}---opencard_pass_rate:{}    opencard_pass_rate_threshold:{}'.format(self.product, opencard_pass_rate, self.config[
                    'opencard_pass_rate_threshold']))
        # if reloan_pass_rate < self.config['reloan_pass_rate_threshold']:
        #     self.except_handler.handle(
        #         msg='{}---reloan_pass_rate:{}    reloan_pass_rate_threshold:{}'.format(self.product, opencard_pass_rate, self.config[
        #             'reloan_pass_rate_threshold']))
        if diversion_pass_rate < self.config['diversion_pass_rate_threshold']:
            self.except_handler.handle(
                msg='{}---diversion_pass_rate:{}    diversion_pass_rate_threshold:{}'.format(self.product, opencard_pass_rate, self.config[
                    'diversion_pass_rate_threshold']))

    def job2(self):
        pass_rate_dict = self.pass_ratio()
        # 当前时间所在天数通过率
        now_str = datetime.now().strftime('%Y-%m-%d')
        opencard_pass_rate = pass_rate_dict.loc[now_str, 'opencard_pass_rate']
        reloan_pass_rate = pass_rate_dict.loc[now_str, 'reloan_pass_rate']
        diversion_pass_rate = pass_rate_dict.loc[now_str, 'diversion_pass_rate']
        if opencard_pass_rate < self.config['opencard_pass_rate_threshold']:
            self.except_handler.handle(
                msg='{}---opencard_pass_rate:{}    opencard_pass_rate_threshold:{}'.format(self.product, opencard_pass_rate, self.config[
                    'opencard_pass_rate_threshold']))
        # if reloan_pass_rate < self.config['reloan_pass_rate_threshold']:
        #     self.except_handler.handle(
        #         msg='{}---reloan_pass_rate:{}    reloan_pass_rate_threshold:{}'.format(self.product, opencard_pass_rate, self.config[
        #             'reloan_pass_rate_threshold']))
        if diversion_pass_rate < self.config['diversion_pass_rate_threshold']:
            self.except_handler.handle(
                msg='{}---diversion_pass_rate:{}    diversion_pass_rate_threshold:{}'.format(self.product, opencard_pass_rate, self.config[
                    'diversion_pass_rate_threshold']))


# class CtlPassMonitor(BasePassMonitor):
#     def __init__(self, config_file='./config/ctlConfig.yaml', product='ctl'):
#         super(CtlPassMonitor, self).__init__(config_file, product)
#
#     def pass_ratio(self):
#         sql = '''select DATE(ml_res.create_time) as create_time_str,sum(if(j_res.result='pass',1,0))/count(-1) as pass_rate
#                 from prod_ctl_mlx.machine_learning_risk_model_results as ml_res
#                 join prod_ctl_mlx.id_relations as ir
#                 on ml_res.app_id = ir.app_id
#                 join prod_ctl_mlx.app_judge_results as j_res
#                  on ir.app_id = j_res.app_id
#                 where j_res.judge_id='finalcheck_judger'
#                  and ir.source ='RELOAN'
#                 and ml_res.create_time>date_add(current_date(), interval -1 day)
#                 group by DATE(ml_res.create_time);'''
#         df_tmp = pd.DataFrame(self.mysql_risk.query(sql))
#         df_tmp['create_time_str'] = df_tmp['create_time_str'].apply(lambda x: str(x))
#         df_tmp = df_tmp.set_index(['create_time_str'])
#         return df_tmp
#
#     def job1(self):
#         logging.info('{} end handle pass_monitor job1!'.format(self.product))
#         pass_rate_dict = self.pass_ratio()
#         # 当前时间前一天所在天数通过率
#         yestoday_str = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
#         reloan_pass_rate = pass_rate_dict.loc[yestoday_str, 'pass_rate']
#         if reloan_pass_rate < self.config['reloan_pass_rate_threshold']:
#             self.except_handler.handle(msg='{}---reloan_pass_rate:{}    reloan_pass_rate_threshold:{}'.format(self.product, reloan_pass_rate,self.config['reloan_pass_rate_threshold']))
#         logging.info('{} end handle pass_monitor job1!'.format(self.product))
#
#
#     def job2(self):
#         logging.info('{} end handle pass_monitor job2!'.format(self.product))
#         pass_rate_dict = self.pass_ratio()
#         # 当前时间所在天数通过率
#         now_str = datetime.now().strftime('%Y-%m-%d')
#         reloan_pass_rate = pass_rate_dict.loc[now_str, 'pass_rate']
#         if reloan_pass_rate < self.config['reloan_pass_rate_threshold']:
#             self.except_handler.handle(msg='{}---reloan_pass_rate:{}    reloan_pass_rate_threshold:{}'.format(self.product, reloan_pass_rate,self.config['reloan_pass_rate_threshold']))
#         logging.info('{} end handle pass_monitor job2!'.format(self.product))


if __name__ == '__main__':
    ltvPassMonitor = LtvPassMonitor()
    omegaPassMonitor = OmegaPassMonitor()
    jdxPassMonitor = JdxPassMonitor()
    ctlPassMonitor = CtlPassMonitor()
    schedule.every().day.at("09:00").do(ltvPassMonitor.job1)
    schedule.every().day.at("18:00").do(ltvPassMonitor.job2)
    schedule.every().day.at("09:00").do(omegaPassMonitor.job1)
    schedule.every().day.at("18:00").do(omegaPassMonitor.job2)
    schedule.every().day.at("09:00").do(jdxPassMonitor.job1)
    schedule.every().day.at("18:00").do(jdxPassMonitor.job2)
    schedule.every().day.at("09:00").do(ctlPassMonitor.job1)
    schedule.every().day.at("18:00").do(ctlPassMonitor.job2)

    while True:
        schedule.run_pending()
        time.sleep(1)
