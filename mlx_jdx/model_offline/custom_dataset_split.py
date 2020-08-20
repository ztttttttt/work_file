from sklearn.utils import shuffle


def train_test_split_by_time(df_data_lbl, app_time_attr, train_start_time=None, train_end_time=None,
                             test_start_time=None, test_end_time=None,
                             slted_cols=None, label_col='label'):
    df_sorted = df_data_lbl.sort_values(by=app_time_attr)

    df_train = (df_sorted.loc[((df_sorted[app_time_attr] >= train_start_time) &
                               (df_sorted[app_time_attr] < train_end_time)), :]).copy()

    df_test = (df_sorted.loc[((df_sorted[app_time_attr] >= test_start_time) &
                              (df_sorted[app_time_attr] < test_end_time)), :]).copy()
    if slted_cols is None:
        X_train = df_train.drop('label', axis=1).copy()
        X_test = df_test.drop('label', axis=1).copy()
    else:
        X_train = df_train[slted_cols].copy()
        X_test = df_test[slted_cols].copy()

    y_train = df_train.loc[:, label_col]
    y_test = df_test.loc[:, label_col]

    return X_train, X_test, y_train, y_test, df_train, df_test


def train_test_split_random(df_data_lbl, test_size=0.25, slted_cols=None, random_state=0, label_col='label'):
    shed_df = shuffle(df_data_lbl, random_state=random_state)

    if type(test_size) == int:
        assert test_size < shed_df.shape[0]
        df_train = shed_df.iloc[:-test_size, :].copy()
        df_test = shed_df.iloc[-test_size:, :].copy()
    elif type(test_size) == float:
        assert 0 < test_size < 1
        test_st_indx = int(shed_df.shape[0] * test_size)  # test set start index
        df_train = shed_df.iloc[:-test_st_indx, :].copy()
        df_test = shed_df.iloc[-test_st_indx:, :].copy()
    else:
        raise Exception('test size type wrong')

    if slted_cols is None:
        X_train = df_train.drop('label', axis=1).copy()
        X_test = df_test.drop('label', axis=1).copy()
    else:
        X_train = df_train[slted_cols].copy()
        X_test = df_test[slted_cols].copy()

    y_train = df_train.loc[:, label_col]
    y_test = df_test.loc[:, label_col]

    return X_train, X_test, y_train, y_test, df_train, df_test
