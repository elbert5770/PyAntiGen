import pandas as pd
import os


def load_no_data(replicate, data_path):
    data_dict = {}  
    return data_dict


def load_ad_data(replicate, data_path):
    def reshape_df(df):
        return pd.concat([
            df[['time', 'B1']].rename(columns={'B1': 'B'}),
            df[['time', 'B2']].rename(columns={'B2': 'B'}),
            df[['time', 'B3']].rename(columns={'B3': 'B'})
        ], ignore_index=True)

    treatment_path = os.path.join(data_path, 'ADneg.csv')
    df_neg = pd.read_csv(treatment_path)
    df_neg_early = reshape_df(df_neg[df_neg['Treatment'] == 'Early'])
    df_neg_late = reshape_df(df_neg[df_neg['Treatment'] == 'Late'])
    treatment_path = os.path.join(data_path, 'ADpos.csv')
    df_pos = pd.read_csv(treatment_path)
    df_pos_early = reshape_df(df_pos[df_pos['Treatment'] == 'Early'])
    df_pos_late = reshape_df(df_pos[df_pos['Treatment'] == 'Late'])
    
    data_dict = {
        "ADneg__Early": df_neg_early,
        "ADneg__Late": df_neg_late,
        "ADpos__Early": df_pos_early,
        "ADpos__Late": df_pos_late,
    }
    return data_dict



