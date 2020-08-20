import logging
import math
from collections import OrderedDict
import numpy as np
from mlx_jdx.dal import jdx_data
from mlx_database.mysql import MySql
from mlx_database.mongo import Mongo
from mlx_utility import config_manager as cm


class CreditAmount:
    def __init__(self, conf_d):
        self.ca = CreditAmountCompiler(conf_d)

    def get_rule_collection_vars(self):
        """
        get all the rule collection variable names used in the judge
        """
        return self.ca.rule_collections

    def get_tag_vars(self):
        """
        get all the tag variable names in the judge
        """
        return self.ca.tags

    def predict(self, app_id, var_dict):
        return self.ca.predict(app_id, var_dict)


class BaseStrategy:
    def __init__(self, strategy, *args, **argw):
        raise Exception("Cann't be instantiated")

    def predict(self, appid, score):
        pass


class BinStrategy(BaseStrategy):
    def __init__(self, strategy, *args, **argw):
        self.name = strategy.get('name')
        self.model_path = strategy.get('model_path')
        self.match = strategy.get('match')
        self.params = strategy.get('params')
        self.output = strategy.get('output')
        self.prob = strategy.get('prob')
        self.is_setup = strategy.get('is_setup')
        self.priority = strategy.get('priority')

        self.mysql_mlx_client = MySql(**cm.config['mysql_jd_cl'])
        self.mongo_mlx_client = Mongo(**cm.config['mongo_jd_cl'])
        self.original_data_access = jdx_data.DerivativeProdData(client=self.mongo_mlx_client)
        self.strategy_result_access = jdx_data.CreditAmountStragetyResult(client=self.mysql_mlx_client)
        self._trans_params()
    
    def is_match(self, **params):
        return eval(self.match.format(**params))

    def _save(self, appid, credit_amount, strategy_name, is_judged):
        self.saver.save(appid, credit_amount, strategy_name, is_judged)

    def _trans_params(self):
        for k in self.params.keys():
            self.params[k] = eval(self.params[k])
    
    def _gen_model_result(self, predit_y, segment_bins):
        agg_pred_amount = OrderedDict()
        proba_dict_res = OrderedDict()
        for bin_ in segment_bins:
            proba_dict_res[bin_] = predit_y
        return proba_dict_res

    def predict_save(self, appid, var_dict):
        credit_amount = self.predict(appid, var_dict)
        thresholds = np.array(self.params['threshold_bins'])
        segment_bins = np.array(self.params['amount_bins'])
        target = var_dict['target']
        score = var_dict['score']
        proba_dict_res = self._gen_model_result(score, segment_bins)
        threshold_dict = OrderedDict(zip(segment_bins, thresholds))
        # save credit_amount to mongo
        col_prefix = ""
        if target != "jd":
            col_prefix = "_" + target.upper()
        self.original_data_access.save_credit_amount(appid.upper(), credit_amount, prefix=col_prefix)
        self.strategy_result_access.save(appid, credit_amount, self.name, '1', thresholds, segment_bins)
        return credit_amount

    def predict(self, appid, var_dict):
        '''predictive amount for proba_arr samples'''
        thresholds = np.array(self.params['threshold_bins'])
        segment_bins = np.array(self.params['amount_bins'])
        score = var_dict['score']
        proba_vector_r = np.array(len(segment_bins) * [score])

        # amount above threshold
        above_th_amount = segment_bins[proba_vector_r >= thresholds]
        if above_th_amount.shape[0] < 1:  # there is no amount above threshold,return default amount: 0
            amount = 0
        else:
            amount = above_th_amount.max()
        return int(amount)


class CreditAmountCompiler:
    def __init__(self, conf_d):
        self.id = conf_d['id']
        self.version = conf_d['version']
        self.parames_required = conf_d.get('parames_required', [])
        self.models = {}
        self.strategies = self._load_strategies(conf_d['strategies'])
        # self._check_prob()

    def _check_prob(self):
        self.sum_prob = sum([strategy.prob for strategy in self.strategies.values() if strategy.is_setup is True and strategy.match is None])
        if self.sum_prob > 1 or self.sum_prob < 0:
            raise Exception("The sum of the probabilities of strategies must between 0 and 1. sum_prob:{}".format(self.sum_prob))
        self.strategies['default'].prob = max(1 - self.sum_prob, 0)

    def _get_class_by_type(self, strategy_type):
        if strategy_type == 'bins':
            return BinStrategy
        else:
            return BaseStrategy

    def _load_strategies(self, dict_strategies):
        strategies = {strategy['name']: self._get_class_by_type(strategy['type'])(strategy) for strategy in dict_strategies if strategy['is_setup'] is True}
        return strategies

    def _choose_one(self, matched_strategies):
        # default_prob = 1 - self.sum_prob
        choice_i = np.random.choice(1000)
        sum_tmp = 0
        for strategy in matched_strategies:
            sum_tmp += strategy.prob * 1000
            if choice_i < sum_tmp:
                return strategy.name
        return None

    def _match_one(self, **params):
        is_matched = False
        matched_priority = math.inf
        matched_strategies = []
        for strategy in sorted(self.strategies.values(), key=lambda obj: obj.priority):
            if strategy.is_match(**params):
                if not is_matched:
                    is_matched = True
                    matched_priority = strategy.priority
                else:
                    if matched_priority < strategy.priority:
                        break 
                matched_strategies.append(strategy)
        return matched_strategies
    
    def predict(self, appid, params):
        d_parames_required = {p_name: params.get(p_name) for p_name in self.parames_required}
        matched_strategies = self._match_one(**d_parames_required)

        if not matched_strategies:
            raise Exception("No Strategy matched!")

        choose_strategy = self._choose_one(matched_strategies)
        credit_results = {choose_strategy: self.strategies[choose_strategy].predict_save(appid, d_parames_required)}
        return choose_strategy, credit_results[choose_strategy]
