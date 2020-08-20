from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.pipeline import Pipeline
from omega.omega_model.feature_encoding import FeaturesEncoding

from mlo_utility import config_manager as cm
cm.setup_config('./config.yaml')


def get_useful_col(train, classifier_):
    print("----get useful columns start----")

    classifier = classifier_

    flow = [('feature_enc', FeaturesEncoding()),
            ('variance', VarianceThreshold(1e-7)),
            ('feature_selection', None),
            ('scale', MinMaxScaler()),
            ('clf', classifier['model'])]
    pipeline = Pipeline(flow)

    clf = GridSearchCV(pipeline, param_grid=classifier['param'], cv=3, scoring='roc_auc', verbose=1, n_jobs=6)
    clf.fit(train.drop('label_class', axis=1), train['label_class'])
    print(clf.best_estimator_.get_params()['steps'])
    print('train cv auc:')
    print(clf.best_score_)

    clf_best = clf.best_estimator_
    clf_best.fit(train.drop('label_class', axis=1), train['label_class'])

    features = clf_best.steps[-5][1].useful_cols[clf_best.steps[-4][1].get_support()][clf_best.steps[-3][1].get_support()]
    if classifier['model_name'] == 'logistic':
        best_features = pd.Series(clf_best.steps[-1][1].coef_[0])
    else:
        best_features = pd.Series(clf_best.steps[-1][1].feature_importances_)
    best_features.index = features
    # print(best_features.sort_values(ascending=False)[:30])

    # useful_col = best_features[best_features > 0].index.tolist()
    useful_col = best_features.sort_values(ascending=False)[:50].index.tolist()
    print("----get useful columns end----")

    return useful_col
