import pandas as pd
import numpy as np
from functools import partial

def map_type(convert_type,x):
    if pd.isnull(x):
        return x
    else:
        if convert_type=='string':
            return str(x)
        elif convert_type=='bool':
            return bool(x)
        else:
            print('type wrong!!!')

def raw_data_convert(df,col_types_df):
    out_dict={}
    for col_name,col_type in col_types_df.values:
        if col_name not in df.columns:
            out_dict[col_name] = np.full((df.shape[0]), np.nan)
        else:
            if col_type == 'float' or col_type == 'int':
                # convert col to number type, 'coerce' will set the col to nan if cannot convert
                out_dict[col_name] = pd.to_numeric(df[col_name], errors='coerce')

            elif col_type == 'string' or col_type == 'bool':
                partial_f = partial(map_type,col_type)
                out_dict[col_name] = df[col_name].map(partial_f)

            else:  # unknown data type,set a default value
                print('wring!!!, get unknown column:{},type:{}'.format(col_name,col_type))
                out_dict[col_name] = np.full((df.shape[0]), 0)
    df_out = pd.DataFrame(out_dict,columns=col_types_df.iloc[:,0])     
    return df_out