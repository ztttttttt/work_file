import pickle

class RiskModel(object):
    '''
    Risk control Model that control pass rate according to category_threshold
    '''

    def __init__(self, estor_path, positive_class_index=1):
        self.positive_class_index = positive_class_index

        # load estimator
        estor_dk = self.__load_estimator_dict(estor_path)
        self.estimator = estor_dk['model']
        #columns used in estimator
        self.fields = self.estimator.get_params()['enum'].clean_col_names
        #model info
        self.model_info = estor_dk['model_info']

    def __load_estimator_dict(self, file_path):
        with open(file_path, 'rb') as f:
            pk_obj = pickle.load(f)
        return pk_obj

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
        return positive_proba
