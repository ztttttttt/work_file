# -*- encoding: utf-8 -*-

import pandas as pd
from matplotlib.cbook import flatten
from mlx_utility.parallel import parallel_dataframe




class FillWithIndicator:
    def __init__(self):
        pass

    def compute_transform(self, df, num_partitions=8, num_cores=8, split_axis=0):
        return parallel_dataframe(map_indicator, df, num_partitions, num_cores, split_axis)

    def transform(self, df):
        assert df.ndim == 2, 'Input shape should be two dimension!!!'

        return map_indicator(df)


def map_indicator(df):
    mapped_df = df.apply(do_map_indicator, axis=1)
    mapped_columns = mapped_df.columns.map(lambda x: [x + '_idt', x])

    new_val = list(map(lambda x: list(flatten(x)), mapped_df.values))
    new_col = list(flatten(mapped_columns))

    output_df = pd.DataFrame(new_val, columns=new_col, index=df.index)

    return output_df


def do_map_indicator(x):
    return list(map(lambda n: [1, n] if pd.notnull(n) else [0, 0], x))
