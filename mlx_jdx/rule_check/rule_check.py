# -*- coding:utf-8 -*-
from abc import abstractmethod
from datetime import datetime, timedelta
import logging
import json
import random
import traceback
import numpy as np
from mlx_jdx.dal.jdx_data import DerivativeProdData, JDRulesData, RuleCheckResultsData


WITHDRAW_TO_OC_CODE = {
    'W001': 'M1Q1',
    'W002': 'M2P1',
    'W003': 'M5T1',
    'W004': 'M5T2',
    'W005': 'M0A1',
    'W006': 'M0A2',
    'W008': 'M5T8'
}


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
        derivative = DerivativeProdData(mongo_client)
        derivative_datas = derivative.get_data_by_appid(app_id)
        if derivative_datas:
            data = dict(data, **derivative_datas)

    return data


def get_group(group_tags, rule_type, rule_check, user_id):
    """

    Return
    ------
    cg_tag: group tag
    is_cg: is control group
    """
    default_tag = group_tags['default']['tag']
    # if rule_type == 'w':
    #     group_tag = rule_check.get_group_tag_by_user_id(user_id)
    #     if not group_tag:
    #         group_tag = default_tag
    #     return group_tag, not group_tag == default_tag
    cg_conf = group_tags.get('control_group', None)
    if cg_conf:
        cg_tag = cg_conf['tag']
        cg_prop = cg_conf['prob']
        max_num = cg_conf['max_num_per_day']
        start_date = datetime.now().date()
        end_date = start_date + timedelta(1)
        if max_num > 0:
            t_num = rule_check.get_cg_count_by_date_tag(start_date, end_date, cg_tag)
            logging.info('t_num:{}, max_num:{}'.format(t_num, max_num))
            if max_num < t_num:
                return default_tag, False
        prop = random.random()
        logging.info('prop:{}, cg_prop:{}'.format(prop, cg_prop))
        if cg_prop > prop:
            return cg_tag, True
    return default_tag, False


def actual_hit_but_pass_rule_code(hit_rule_code, actual_hit_rule_code):
    try:
        return list(set(json.loads(actual_hit_rule_code)) - set(json.loads(hit_rule_code)))
    except BaseException:
        return []


def rule_check(app_id, category_id, server_name, mongo_client, ml_jd_client, rule_code_p=None, **argw):
    rule_obj = JDRulesData(ml_jd_client)
    rule_check = RuleCheckResultsData(ml_jd_client)

    data = fetch_data(app_id, mongo_client, rule_code_p)

    result = True
    rule_type = argw.pop('rule_type', None)
    group_tags = argw.pop('group_tags', None)

    user_id = data.get('X_UserId')
    group_tag, is_cg = get_group(group_tags, rule_type, rule_check, user_id)

    if not rule_code_p:
        rules_datas = rule_obj.get_rules_by_server(server_name, rule_type, category_id)
    else:
        rules_datas = rule_obj.get_rule_by_rule_code(rule_code_p, rule_type, category_id)
    if not rules_datas:
        logging.info('app_id rule check over(no rules), result:{result}'.format(result=result))
        return result, group_tag, json.dumps([]), user_id

    hit_rule = []
    actual_hit_rule = []

    open_card_result = rule_check.get_open_card_rule_check_result(user_id)
    if open_card_result:
        hit_rule_code = open_card_result['hit_rule_code']
        actual_hit_rule_code = open_card_result['actual_hit_rule_code']
    else:
        hit_rule_code = "[]"
        actual_hit_rule_code = "[]"

    for rule in rules_datas:
        rule_code = rule['rule_code']
        pass_prob = rule['pass_prob']
        prob_gen_manner = rule['prob_gen_manner']
        prob_params = rule['prob_params']
        check_model = rule['check_model']
        model_params = rule['model_params']
        model = get_rule_check(check_model)
        prob_generator = get_prob_generator(prob_gen_manner)
        # if check_result is true represents the application hit the rule
        check_result = model(app_id, data, rule_code, model_params, **argw).do_check()

        if check_result:
            hit_rule.append(rule_code)
            actual_hit_rule.append(rule_code)

        if rule_type == 'w' and WITHDRAW_TO_OC_CODE.get(rule_code) in actual_hit_but_pass_rule_code(hit_rule_code, actual_hit_rule_code) and rule_code in hit_rule:
            hit_rule.remove(rule_code)
        elif check_result and pass_prob > 0:
            prob = prob_generator(prob_gen_manner, prob_params).gen_prob()
            logging.info('app_id:{app_id}, category_id:{category_id}, rule_code:{rule_code}, pass_prob:{pass_prob}, gen_prob:{prob}'.format(app_id=app_id, category_id=category_id, rule_code=rule_code, pass_prob=pass_prob, prob=prob))
            if (rule_type == 'w' and prob < pass_prob)\
               or (rule_type == 'c' and rule_code in actual_hit_but_pass_rule_code(hit_rule_code, actual_hit_rule_code))\
               or (rule_type == 'c' and prob < pass_prob and not open_card_result):
                hit_rule.remove(rule_code)

    result = 0 if hit_rule and not is_cg else 1
    actual_result = 0 if actual_hit_rule else 1

    rule_check_result = dict()
    hit_rule_str = json.dumps(hit_rule)
    rule_check_result['hit_rule'] = hit_rule_str
    rule_check_result['actual_hit_rule'] = json.dumps(actual_hit_rule)
    rule_check_result['result'] = result
    rule_check_result['actual_result'] = actual_result
    rule_check_result['user_id'] = user_id
    rule_check_result['group_tag'] = group_tag
    rule_check_result['rule_type'] = rule_type
    rule_check_result['category_id'] = category_id
    rule_check.save(app_id, rule_check_result)

    logging.info('app_id:{app_id} rule check over, result:{result}'.format(app_id=app_id, result=result))
    return result, group_tag, hit_rule_str, user_id


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
    def __init__(self, app_id, datas, rule_code, rule_params, **argw):
        self.datas = datas
        self.rule_params = rule_params
        self.rule_code = rule_code
        self.app_id = app_id

    def do_check(self):
        return False


class RuleCheckDefault(RuleCheckBase):
    def __init__(self, app_id, datas, rule_code, rule_params, **argw):
        super(RuleCheckDefault, self).__init__(app_id, datas, rule_code, rule_params, **argw)

    def __condition_exec(self, condition):
        result = False
        field = condition['field']
        f_value = self.datas.get(field)
        if not f_value:
            return result
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
