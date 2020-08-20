import numpy as np
import pandas as pd
from scipy.stats import beta
from mlx_jdx.model_online.credit_make_helper import make_amount


class BetaCreditGiver:

    def __init__(self, good_prob, pass_rate, credits, alpha, beta, rule_reject, both_reject_ratio_in_rule):

        self.good_prob = good_prob
        self.pass_rate = pass_rate
        self.credits = credits
        self.alpha = alpha
        self.beta = beta
        self.rule_reject = rule_reject
        self.both_reject_ratio_in_rule = both_reject_ratio_in_rule
        self.model_pass_rate = self.thresh_calc_with_rule()

    def beta_redistribute_by_prob(self, min, avg):
        prob = self.good_prob
        prob_sorted = pd.DataFrame(pd.Series(prob).sort_values(ascending=True))
        prob_sorted.columns = ['prob']
        prob_sorted.loc[:, 'percentile'] = range(0, prob_sorted.shape[0])
        prob_sorted.loc[:, 'percentile'] = prob_sorted['percentile'] / (prob_sorted.shape[0] + 0.0)

        passed = prob_sorted.loc[prob_sorted['percentile'] >= 1 - self.model_pass_rate].copy()
        passed.loc[:, 'normed_percentile'] = (passed['percentile'] - passed['percentile'].min()) / (
                passed['percentile'].max() - passed['percentile'].min())

        passed.loc[:, 'credit'] = passed['normed_percentile'].map(
            lambda x: self.find_beta_credit(self.alpha, self.beta, x, min + 1, avg))

        threshold = []
        for credit in self.credits:
            threshold.append(passed.loc[np.abs(passed['credit'] - credit).argmin(), 'prob'])

        passed.loc[:, 'credit_floored'] = passed['credit'].map(lambda x: self.floor_to_credit(x, self.credits))

        rejected = prob_sorted.loc[prob_sorted['percentile'] < 1 - self.model_pass_rate]
        rejected_thresh = rejected.quantile(np.linspace(0.1, 0.9, 9))
        prob_sorted['credit'] = 0
        prob_sorted.loc[passed.index, 'credit'] = passed['credit_floored']

        return threshold, prob_sorted['credit'], rejected_thresh

    def find_beta_credit(self, a, b, pct, min, avg):
        #use expectation to calculate max value
        max = min + (avg - min) * (a + b) / a
        credit = min + beta.ppf(pct, a, b) * (max - min)
        return credit

    def floor_to_credit(self, x, credits):
        for i, credit in enumerate(credits):
            if x < credit:
                floored_x = credits[i - 1]
                break
            elif x > credits[-1]:
                floored_x = credits[-1]
        return floored_x

    def thresh_calc_with_rule(self):
        reject = 1 - self.pass_rate
        model_reject = reject - self.rule_reject + self.both_reject_ratio_in_rule * self.rule_reject
        model_pass = 1 - model_reject
        return model_pass


class GaussianCreditGiver:
    '''
    given model predictive probabilities, pass rate, amount bins,
    and gaussian distribution params:  mu and sigma.
    return thresholds of each bin, and given credit for each 'probability'
    '''

    def mpm_mapping(self, pred_probas, ps_rt, bins, mu, sigma):
        normed_gaussian_ps_rt = self.__gaussian_pass_rate(ps_rt, bins, mu, sigma)

        # cumulative gaussian pass rate
        cum_ps_rt = np.cumsum(normed_gaussian_ps_rt[::-1])[::-1]

        rate_val = list(map(lambda x: 100 - x * 100, cum_ps_rt))
        # thresholds of bins
        thresholds = np.percentile(pred_probas, rate_val)

        given_credit_arr = []
        for proba in pred_probas:
            proba_vector = len(bins) * [proba]
            pred_amount = make_amount(proba_vector, thresholds, bins)
            given_credit_arr.append(pred_amount)
        return thresholds, np.array(given_credit_arr)

    def __gaussian_pass_rate(self, ps_rt, bins, mu, sigma):
        '''
        bins: input value
        mu: mean value
        sigma: std value
        '''
        gaussians = 1.0 / (sigma * np.sqrt(2.0 * np.pi)) * np.exp(- (np.array(bins) - mu) ** 2.0 / (2.0 * sigma ** 2.0))

        normed_gaussian_ps_rt = (ps_rt * 1.0 / sum(gaussians)) * gaussians
        return normed_gaussian_ps_rt
