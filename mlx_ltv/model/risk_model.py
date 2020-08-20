import pickle


class RiskModel(object):
    '''
    Risk control Model that control pass rate according to category_threshold
    '''

    def __init__(self, estor_path, positive_class_index=1):
        self.positive_class_index = positive_class_index

        # load estimator
        estor__dk = self.__load_estimator_dict(estor_path)
        self.estimator = estor__dk['model']
        self.fields = self.estimator.get_params()['enum'].clean_col_names

        self.estimator_name = estor__dk['model_name']
        self.estimator_train_date = estor__dk['train_date_score']['date']
        self.train_score = estor__dk['train_date_score']['score']
        self.data_label = estor__dk['data_label']

        self.credit_field = estor__dk['credit_field']
        self.principal_field = estor__dk['principal_field']

    def __load_estimator_dict(self, file_path):
        estor_dict = self.__load_persistence_file(file_path)
        return estor_dict

    def __load_persistence_file(self, file_path):
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
