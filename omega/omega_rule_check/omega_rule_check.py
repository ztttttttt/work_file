# -*- coding:utf-8 -*-
from abc import abstractmethod
import logging
import json
import traceback
import numpy as np
from asteval import Interpreter
from datetime import datetime, timedelta
from omega.omega_dal.data_source import DerivativeProdData, OmegaRuleCheckResultData, OmegaRulesData, RiskPreauditData


def get_prob_generator(prob_gen_manner):
    return GenProbDefault


def get_rule_check(check_model):
    if check_model is None or check_model.strip() == '':
        return RuleCheckDefault
    else:
        logging.warning('Check model error, check_model:{model} is undefined!'.format(model=check_model))
        return RuleCheckBase


def fetch_data(app_id, mongo_client, rule_code=None):
    data = {}    # merge other datas to one dict, now just have derivables
    if not rule_code:

        try:
            derivative = DerivativeProdData(mongo_client)
            derivative_datas = derivative.get_data_by_appid(app_id)
            if derivative_datas:
                data = dict(data, **derivative_datas)
        except:
            logging.error('get data error, app_id: {app_id}, info:{err_info}.'.format(app_id=app_id, err_info=traceback.format_exc()))
            data = {}

    return data


def get_group_tag(user_id, group_tags, server_name, data, rule_check_data):
    aeval = Interpreter()
    default_tag = group_tags['default']['tag']

    # if server_name == 'omega_withdraw':
    #    if not user_id:
    #        group_tag = default_tag
    #        return group_tag, False
    #    group_tag_info = rule_check_data.get_group_tag(user_id)
    #    group_tag = group_tag_info[-1]['group_tag'] if group_tag_info else None
    #    if not group_tag:
    #        group_tag = default_tag
    #    return group_tag, not group_tag == default_tag

    if 'control_group' in group_tags:
        group_tag = group_tags['control_group']['tag']
        prob = group_tags['control_group']['prob']
        max_num = group_tags['control_group']['max_num_per_day']
        not_cgs = group_tags['control_group']['not_cg_condition']
        
        if max_num > 0:
            start_time = datetime.now().date()
            end_time = start_time + timedelta(days=1)
            nonce_num = rule_check_data.get_cg_count_by_date(start_time, end_time, group_tag)
            cg_num = nonce_num[0]['cg_num'] if nonce_num else 0
            logging.info('nonce num:{}, max num:{}'.format(cg_num, max_num))
            if cg_num >= max_num:
                return default_tag, False
            
            res = False
            for not_c in not_cgs:
                result = True
                for kk, vv in not_c.items():
                    value = data.get(kk)
                    if value:
                        aeval.symtable['VALUE'] = value
                        is_match = aeval(vv)
                        result = result and is_match  
                    else:
                        result = False
                        continue
                res = res or result
                
            rand_prob = np.random.rand(1)[0]
            logging.info('rand prob: {}, control group prob:{}'.format(rand_prob, prob))
            if (not res) and rand_prob <= prob:
                return group_tag, True

    return default_tag, False


def rule_check(app_id, server_name, mongo_client, ml_o_client, rule_code_p=None, **argw):
    rule_obj = OmegaRulesData(ml_o_client)
    rule_check_data = OmegaRuleCheckResultData(ml_o_client)

    risk_client = argw.pop('mysql_risk', None)
    group_tags = argw.pop('group_tags', None)

    pre_audit = RiskPreauditData(risk_client)

    data = fetch_data(app_id, mongo_client, rule_code_p)
    omega_channel = int(data.get('OMEGA_CHANNEL', -2))

    # this 'OMEGA_USER_ID' must be upper, because in the data source file, we convert all keys to upper
    user_id = data.get('OMEGA_USER_ID')

    group_tag, is_cg = get_group_tag(user_id, group_tags, server_name, data, rule_check_data)

    result = False

    if not rule_code_p:
        rules_datas = rule_obj.get_rules_by_server(server_name, omega_channel)
    else:
        rules_datas = rule_obj.get_rule_by_rule_code(rule_code_p, omega_channel)
    if not rules_datas:
        logging.info('app_id rule check over(no rules), result:{result}'.format(result=result))
        return result, group_tag

    hit_rule = []
    actual_hit_rule = []

    for rule in rules_datas:
        rule_code = rule['rule_code']
        pass_prob = rule['pass_prob']
        prob_gen_manner = rule['prob_gen_manner']
        prob_params = rule['prob_params']
        check_model = rule['check_model']
        model_params = rule['model_params']
        convert_type_to = rule['convert_type_to']

        model = get_rule_check(check_model)
        prob_generator = get_prob_generator(prob_gen_manner)
        # if check_result is true represents the application hit the rule
        check_result = model(app_id, data, rule_code, model_params, convert_type_to, **argw).do_check()

        if check_result:
            hit_rule.append(rule_code)
            actual_hit_rule.append(rule_code)

        if check_result and pass_prob > 0:
            prob = prob_generator(prob_gen_manner, prob_params).gen_prob()
            if prob < pass_prob:
                hit_rule.remove(rule_code)

    result = 0 if hit_rule and not is_cg else 1
    actual_result = 0 if actual_hit_rule else 1

    rule_check_result = dict()
    rule_check_result['hit_rule'] = json.dumps(hit_rule)
    rule_check_result['actual_hit_rule'] = json.dumps(actual_hit_rule)
    rule_check_result['result'] = result
    rule_check_result['actual_result'] = actual_result
    rule_check_result['service'] = server_name
    rule_check_result['omega_channel'] = omega_channel
    rule_check_result['user_id'] = user_id
    rule_check_result['group_tag'] = group_tag
    rule_check_data.save(app_id, rule_check_result)

    if hit_rule:
        pre_audit.update_result_by_id(app_id, ",".join(hit_rule))

    logging.info('app_id:{app_id} rule check over, result:{result}'.format(app_id=app_id, result=result))
    return result, group_tag


class GenProbBase(object):
    def __init__(self, prob_gen_manner, prob_params):
        self.prob_gen_manner = prob_gen_manner
        self.prob_params = prob_params

    @abstractmethod
    def gen_prob(self):
        pass


class GenProbDefault(GenProbBase):
    def __init__(self, prob_gen_manner, prob_params):
        super(GenProbDefault, self).__init__(prob_gen_manner, prob_params)

    def gen_prob(self):
        return np.random.rand(1)[0]


class RuleCheckBase(object):
    def __init__(self, app_id, datas, rule_code, rule_params, convert_type, **argw):
        self.datas = datas
        self.rule_params = rule_params
        self.rule_code = rule_code
        self.app_id = app_id
        self.convert = convert_type

    def do_check(self):
        return False


class RuleCheckDefault(RuleCheckBase):
    def __init__(self, app_id, datas, rule_code, rule_params, convert_type, **argw):
        super(RuleCheckDefault, self).__init__(app_id, datas, rule_code, rule_params, convert_type, **argw)

    def __condition_exec(self, condition):
        result = False
        field = condition['field']
        f_val = self.datas.get(field.upper())
        if not f_val:
            return result

        f_value = self.__convert_type_to(f_val)

        op = condition['op']
        value = condition['val']
        if op == '==':
            result = f_value == value
        elif op == '>':
            result = f_value > value
        elif op == '>=':
            result = f_value >= value
        elif op == '<':
            result = f_value < value
        elif op == '<=':
            result = f_value <= value
        elif op == '!=':
            result = f_value != value
        elif op == 'in':
            result = f_value in value
        elif op == 'nin':
            result = f_value not in value

        return result

    def __conditions_exec(self, conditions, log_op):
        if log_op == 'and':
            result = True
            for condition in conditions:
                sub_conditions = condition.get('con')
                if sub_conditions:
                    result = self.__conditions_exec(sub_conditions, condition.get('log_op'))
                else:
                    result = result and self.__condition_exec(condition)
        else:
            result = False
            for condition in conditions:
                sub_conditions = condition.get('con')
                if sub_conditions:
                    result = self.__conditions_exec(sub_conditions, condition.get('log_op'))
                else:
                    result = result or self.__condition_exec(condition)
        return result

    def do_check(self):
        """
        check the rules
        ---------
        return value: if the app_id hit the rule then true, else false. So, it's opposite with the man is good or bad.
        """
        result = False   # default is not hit the rule

        try:
            params = json.loads(self.rule_params)
            conditions = params['con']
            log_op = params['log_op']
            result = self.__conditions_exec(conditions, log_op)
        except Exception:
            logging.warning('rule error, rule_code:{rule_code}, error info:{err_info}.'.format(rule_code=self.rule_code, err_info=traceback.format_exc()))
        return result

    def __convert_type_to(self, value):
        if not self.convert:
            return value
        if self.convert == 'int':
            return int(value)
        if self.convert == 'str':
            return str(value)
        if self.convert == 'float':
            return float(value)
