
import numpy as np
import pandas as pd
import re

def match_mobile(uq_col_val):
    res = []
    for x in uq_col_val:
        if type(x) == str:
            if re.match(r'1\d{10}', x):
                res.append(True)
            else:
                res.append(False)

    return np.array(res).all() and len(res) > 0


def determine_column_type(df, enum_cover_rate=0.5):
    common_enum_columns = [
        'X_BD_BlackList_RetCode',
        'X_BD_BlackList_SignType',
        'X_BR_Apply_Code',
        'X_Graph_Mdx_main_province',
        'X_Mobile_ThreeInfo',
        'X_APP_DeviceBaseStation',
        'X_APP_DeviceId',
        'X_MX_raw_data_calls_items_peer_number',
        'X_MX_raw_data_families_family_num',
        'X_MX_raw_data_families_items_short_number',
        'X_XINYAN_Black_code',
        'X_XINYAN_Radar_code',
        'X_MX_raw_report_main_service_service_num',
        'X_MX_raw_report_sms_contact_detail_peer_num',
        'X_JD_BankCardNo',
        'X_JD_MerchantCode',
        'X_JD_WithdrawsAfterSmsCheck_ResultCode',
        'X_JD_Withdraws_DecisionResult',
        'X_JD_Withdraws_ResultCode',
        'X_SZR_Education',
        'X_SZR_EducationType'
        
    ]

    bool_ambigious_set = {False, True, ''}
    meaningless_set = {'', None, 'None', 'Null', 'null', np.nan}
    agg_effective_col_types = []
    for col in df.columns:
        if df[col].count() == 0:  # empty column
            pass
        else:
            if pd.api.types.is_datetime64_dtype(df[col]):  # datetime column
                pass
            elif df[col].dtype == np.bool:  # bool column
                agg_effective_col_types.append((col, 'bool'))

            elif df[col].dtype == np.int:  # int column
                agg_effective_col_types.append((col, 'int'))

            elif df[col].dtype == np.float:  # float column
                agg_effective_col_types.append((col, 'float'))

            elif df[col].dtype == np.object:  # object column that require further analysis
                try:
                    # col can be convert to number
                    uq_col_val = pd.unique(df[col])  # check whether column can be unique
                except Exception as e:
                    pass
                else:
                    if set(uq_col_val).issubset(meaningless_set):  # meaningless column
                        pass
                    else:
                        try:
                            obj_converted = pd.to_numeric(df[col], errors='raise')
                        except:  # enum column
                            if len(uq_col_val) <= df[
                                col].count() * 1.0 * enum_cover_rate:  # length of unique enum column  under threshold
                                agg_effective_col_types.append((col, 'string'))
                        else:
                            # match mobile number
                            if match_mobile(uq_col_val):  # col is mobile number
                                pass
                            else:
                                if ({True}.issubset(set(uq_col_val)) or
                                        {False}.issubset(set(uq_col_val))
                                    or col in common_enum_columns):  # column that match common object type
                                    agg_effective_col_types.append((col, 'string'))
                                else:  # numeric columns that should be checked mannually!!!
                                    print(col)
                                    if obj_converted.dtype == np.int:
                                        agg_effective_col_types.append((col, 'int'))

                                    elif obj_converted.dtype == np.float:
                                        agg_effective_col_types.append((col, 'float'))
                                    else:
                                        print('wrong!!!, column:{},type:{}'.format(col,obj_converted.dtype))
            else:  # unknown type, just skip
                print('unknown type, column:{}'.format(col))
    return agg_effective_col_types