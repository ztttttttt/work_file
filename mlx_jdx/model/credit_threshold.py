import numpy as np
import pandas as pd
from scipy.stats import beta


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

    def beta_redistribute_by_prob(self, min=2000, avg=4000):
        prob = self.good_prob
        prob_sorted = pd.DataFrame(pd.Series(prob).sort_values(ascending=True))
        prob_sorted.columns = ['prob']
        prob_sorted['percentile'] = range(0, prob_sorted.shape[0])
        prob_sorted['percentile'] = prob_sorted['percentile'] / (prob_sorted.shape[0] + 0.0)

        passed = prob_sorted.loc[prob_sorted['percentile'] >= 1 - self.model_pass_rate]
        passed['normed_percentile'] = (passed['percentile'] - passed['percentile'].min()) / (
                passed['percentile'].max() - passed['percentile'].min())

        passed['credit'] = passed['normed_percentile'].map(
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

    def find_beta_credit(self, a, b, pct, min=0, avg=30):
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
