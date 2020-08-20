if __name__ == '__main__':

    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, SelectFromModel
    from sklearn.model_selection import GridSearchCV, train_test_split
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    import numpy as np
    import pandas as pd
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
    import xgboost
    from xgboost import XGBClassifier
    from sklearn.utils import shuffle
    import pickle
    from omega.omega_model.feature_encoding import FeaturesEncoding
    import get_useful_col

    from datetime import datetime, timedelta
    from mlo_utility import config_manager as cm
    cm.setup_config('./config.yaml')

    # train = pd.read_csv('/home/wenna/omega_model/dataset/201804031pd7_traindata_new.csv')
    """
    train = pd.read_pickle('/home/wenna/omega_model/Data/201805151pd7_train_data.dfp')
    train['label_class'] = train['label_class'].map({1: -1, 0: 1})

    if cm.config['train_data_shuffle']:
        train = shuffle(train)

    test = train.iloc[:int(train.shape[0]*0.3)]
    train = train.drop(test.index)

    train.to_csv('/home/wenna/omega_model/dataset/train_201805151pd7_4.csv', index=False)
    test.to_csv('/home/wenna/omega_model/dataset/test_201805151pd7_4.csv', index=False)
    """
    train = pd.read_pickle("/home/wenna/omega_model/dataset/train_201807091pd7.dfp")
    test = pd.read_pickle("/home/wenna/omega_model/dataset/test_201807091pd7.dfp")

    if cm.config['train_data_shuffle']:
        train = pd.read_pickle("/home/wenna/omega_model/dataset/train_201807091pd7_shullfe.dfp")
        test = pd.read_pickle("/home/wenna/omega_model/dataset/test_201807091pd7_shullfe.dfp")

    print('train data shape: ', train.shape)
    print('train label counts: ', '\n', train['label_class'].value_counts())
    print('test data shape: ', test.shape)
    print('test label counts: ', '\n', test['label_class'].value_counts())

    params_lg = {
                'feature_enc__cat_cols_strategy': ['ordinal'],
                'feature_enc__num_cols_strategy': [-2, 'most_frequent'],
                'feature_enc__max_num': [20],
                'scale': [MinMaxScaler(), StandardScaler()],
                'feature_selection': [SelectFromModel(XGBClassifier(n_estimators=100)),
                                      SelectFromModel(ExtraTreesClassifier()), SelectKBest(k='all')],
                'clf__penalty': ['l2'],
                'clf__C': [0.0001, 0.001, 0.01, 0.1],
                'clf__class_weight': ['balanced']
    }

    params_randomforest = {
              'feature_enc__num_cols_strategy': [-2, 'most_frequent'],
              'feature_enc__cat_cols_strategy': ['ordinal'],
              'feature_enc__max_num': [50],
              'feature_selection': [SelectFromModel(ExtraTreesClassifier()), SelectKBest(k='all')],
              'scale': [MinMaxScaler(), StandardScaler()],
              'clf__max_depth': [3, None],
              'clf__n_estimators': [50, 100, 200],
              'clf__max_features': ['sqrt'],
              'clf__class_weight': ['balanced_subsample'],
              'clf__n_jobs': [1],
              }

    params_xgb = {
                'feature_enc__cat_cols_strategy': ['ordinal'],
                'feature_enc__num_cols_strategy': ['most_frequent', 'mean'],
                'feature_enc__max_num': [50],
                'feature_selection': [SelectKBest(k='all'), SelectFromModel(GradientBoostingClassifier())],
                'scale': [StandardScaler()],
                'clf__learning_rate': [0.01],
                'clf__max_depth': [3],
                'clf__min_child_weight': [1],
                'clf__n_estimators': [300],
                'clf__reg_alpha': [0.5],
                'clf__reg_lambda': [0],
                'clf__subsample': [0.5],
                'clf__scale_pos_weight': [1.2],
                'clf__colsample_bytree': [1],
                'clf__gamma': [0],
                'clf__nthread': [1]
    }

    randomforest = {'model_name': 'randomforst', 'model': RandomForestClassifier(), 'param': params_randomforest}
    lg = {'model_name': 'logistic', 'model': LogisticRegression(), 'param': params_lg}
    xgb = {'model_name': 'xgb', 'model': XGBClassifier(), 'param': params_xgb}

    classifier = xgb

    useful_col = get_useful_col.get_useful_col(train, classifier)
    # model = pickle.load(open("/home/wenna/omega_model/model/xgb_2018-02-01_2018-04-07_2.p", "rb"))
    # features = model['features']
    # useful_col = features[features > 0].index.tolist()
    train = train[useful_col+['label_class']]

    flow = [('feature_enc', FeaturesEncoding()),
            ('variance', VarianceThreshold(1e-4)),
            ('feature_selection', None),
            ('scale', MinMaxScaler()),
            ('clf', classifier['model'])]
    pipeline = Pipeline(flow)

    clf = GridSearchCV(pipeline, param_grid=classifier['param'], cv=3, scoring='roc_auc', verbose=1, n_jobs=6)
    clf.fit(train.drop('label_class', axis=1), train['label_class'])
    print(clf.best_estimator_.get_params()['steps'])
    print('train cv auc:')
    print(clf.best_score_)

    start_time = datetime.now()
    clf_best = clf.best_estimator_
    clf_best.fit(train.drop('label_class', axis=1), train['label_class'])
    end_time = datetime.now()
    train_time = (end_time - start_time).seconds
    pred_prob = clf_best.predict_proba(test.drop('label_class', axis=1))[:, 1]
    print('test auc:')
    print(roc_auc_score(test['label_class'], pred_prob))
    print('test pr-auc:')
    print(average_precision_score(test['label_class'], pred_prob))

    best_score = clf.best_score_
    best_model = clf_best
    best_model_name = classifier['model_name']

    features = clf_best.steps[-5][1].useful_cols[clf_best.steps[-4][1].get_support()][clf_best.steps[-3][1].get_support()]
    if classifier['model_name'] == 'logistic':
        best_features = pd.Series(clf_best.steps[-1][1].coef_[0])
    else:
        best_features = pd.Series(clf_best.steps[-1][1].feature_importances_)
    best_features.index = features
    # print(best_features.sort_values(ascending=False)[:30])

    model_to_save = {'model_name': '{}_{}_{}'.format(best_model_name, cm.config['train_start_time'], cm.config['train_end_time']),
                     'model': best_model,
                     'CV_AUC': best_score,
                     'features': best_features.sort_values(ascending=False),
                     'label': cm.config['delay_days'],
                     'train_time(secs)': train_time}

    pickle.dump(model_to_save, open('{}/xgb_{}_{}.p'.format(cm.config['model_save_dir'], cm.config['train_start_time'],
                                                              cm.config['train_end_time']), 'wb'))
