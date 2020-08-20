import numpy as np


class ThresholdController:
    # control first and last pass rate number precision
    EPSILON_LOW_BOUND = -0.01
    EPSILON_UPPER_BOUND = 1.01
    '''
    control thresholds dynamically
    '''

    def __init__(self):
        pass

    def calculate_new_threshold(self, current_pass_rate, current_threshold, pass_rate_bounds, threshold_delta_steps,
                                lowest_threshold, highest_threshold):
        '''
        calculate new category_threshold according to current category_threshold and its category_threshold delta
        :param current_pass_rate: current pass rate
        :param current_threshold: current category_threshold
        :param pass_rate_bounds: an array of different pass rate bounds,its length equals to threshold_delta_steps + 1
        :param threshold_delta_steps: category_threshold delta corresponding to the pass_rate_bounds
        :param lowest_threshold:  the lowest category_threshold that can receive
        :param highest_threshold: the highest category_threshold that can receive
        :return: new category_threshold
        '''
        assert (len(pass_rate_bounds) == len(threshold_delta_steps) + 1), 'length do not match'
        pass_rate_bounds[0] = self.EPSILON_LOW_BOUND
        pass_rate_bounds[-1] = self.EPSILON_UPPER_BOUND

        # category_threshold delta corresponding to current pass rate
        threshold_delta = threshold_delta_steps[np.searchsorted(pass_rate_bounds, current_pass_rate) - 1]
        # update current category_threshold
        new_threshold = float(current_threshold) + threshold_delta

        # ensure category_threshold not go outside bounds
        if new_threshold < lowest_threshold:
            new_threshold = lowest_threshold
        if new_threshold > highest_threshold:
            new_threshold = highest_threshold

        return new_threshold
