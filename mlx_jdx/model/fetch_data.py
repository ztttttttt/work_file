from mlx_database.mongo import Mongo
import pandas as pd
import datetime
import os
from mlx_utility import config_manager as cm

cm.setup_config()

def fetch_data_from_db(start_time, end_time):
    db = Mongo(**cm.config['mongo_config'])
    Collection = db.get_collection(cm.config['collection'])
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)
    df = pd.DataFrame(list(Collection.find({'create_time': {'$gte': start_time, '$lt': end_time}}, {'X_SZR_EntryDate': 0})))
    bad_cols = []
    for col in df.columns:
        if 'X_JD_ActiveCard' in col:
            bad_cols.append(col)
    df = df.drop(bad_cols, axis=1)
    df.loc[:,'_id'] = df['_id'].str.lower()
    df = df.set_index('_id', drop=False)
    return df

def fetch_data_from_file(start_time, end_time, fl_dir='/mnt/Data/New/prod_jdcl/derivables/'):
    date_arr = pd.date_range(start=start_time, end=end_time, freq='D')[:-1].map(lambda x: x.strftime('%Y-%m-%d'))
    agg_data = []
    for date_str in date_arr:
        fl_path = os.path.join(fl_dir, '{}.csv'.format(date_str))
        tmp_d = pd.read_csv(fl_path, encoding='utf-8', thousands=',')
        agg_data.append(tmp_d)
    df = pd.concat(agg_data, axis=0, join='inner')
    bad_cols = []
    for col in df.columns:
        if 'X_JD_ActiveCard' in col:
            bad_cols.append(col)
    df = df.drop(bad_cols, axis=1)
    df.loc[:,'_id'] = df['_id'].str.lower()
    df = df.set_index('_id', drop=False)
    return df


def fetch_label_from_file(label_date, delay_days=3, fl_dir='/mnt/Data/New/prod_jdcl/jdcl_label', label_col='first_overdue_date'):
    train_label = pd.read_csv('{}/{}.csv'.format(fl_dir, label_date))
    train_label['app_id'] = train_label['app_id'].str.lower()
    train_label = train_label[['app_id', label_col, 'withdraw_date']].set_index('app_id')
    train_label['withdraw_date'] = pd.to_datetime(train_label['withdraw_date'])
    train_label = train_label.loc[train_label['withdraw_date']<pd.to_datetime(label_date)-datetime.timedelta(days=delay_days)]
    train_label['label_class'] = 0
    train_label.loc[train_label[label_col] >= delay_days, 'label_class'] = 1
    train_label['label_class'] = train_label['label_class'].map({1: -1, 0: 1})
    return train_label['label_class']

def data_label_combine(data, label):
    shared_index = pd.Index(set(data.index) & set(label.index))
    data_shared = data.loc[shared_index]
    label_shared = label.loc[shared_index]
    return pd.concat([data_shared, label_shared], axis=1, join='inner')