from pathlib import PurePath
import pickle
from mlx_model.base_model import BaseModel


class RiskModel(object):
    '''
    Risk control Model that control pass rate according to category_threshold
    '''

    def __init__(self, estor_path, positive_class_index=1):
        # super(RiskModel, self).__init__(estor_path, positive_class_index)
        model_info = self._load_model_from_file(estor_path)
        self.estimator = model_info['model']
        p = PurePath(estor_path)
        self.estimator_name = model_info['model_name']
        self.data_label = 'fpd3'
        self.positive_class_index = positive_class_index

    def _load_model_from_file(self, f_path):
        with open(f_path, 'rb') as fr:
            return pickle.load(fr)

    def model_predict_in_pipeline(self, input_data_df):
        '''
        :param input_data:  two dimension array of features
        :param category_threshold:  scalar; category_threshold of category where this user is in
        :return: predict probability
        '''
        assert input_data_df.ndim == 2, 'input_data should be two dimension'

        # transform data and make prediction
        pred_probas = self.estimator.predict_proba(input_data_df)
        positive_proba = pred_probas[:, self.positive_class_index][0]
        return float(positive_proba)
