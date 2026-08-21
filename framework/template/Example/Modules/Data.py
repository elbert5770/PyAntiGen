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
        "ADneg_Early": df_neg_early,
        "ADneg_Late": df_neg_late,
        "ADpos_Early": df_pos_early,
        "ADpos_Late": df_pos_late,
    }
    return data_dict


def load_flipflop_data(replicate, data_path):
    """Load the synthetic flip-flop dataset (see data/make_flipflop_data.py).

    Returns log10-scale data: for each treatment a stacked frame of the three
    logB replicate columns, plus a separate sparse frame of the noisy logA
    observations (Early treatment only), keyed "<label>_A". The logA points are
    what break the flip-flop swap symmetry and set the height of the second
    likelihood mode.
    """
    def stack_logB(df):
        return pd.concat([
            df[['time', 'logB1']].rename(columns={'logB1': 'logB'}),
            df[['time', 'logB2']].rename(columns={'logB2': 'logB'}),
            df[['time', 'logB3']].rename(columns={'logB3': 'logB'})
        ], ignore_index=True)

    df = pd.read_csv(os.path.join(data_path, 'Flipflop.csv'))
    data_dict = {}
    for treatment in ('Early', 'Late'):
        sub = df[df['Treatment'] == treatment]
        data_dict[f"Flipflop_{treatment}"] = stack_logB(sub)
        logA = sub[sub['logA'].notna()][['time', 'logA']].reset_index(drop=True)
        if len(logA):
            data_dict[f"Flipflop_{treatment}_A"] = logA
    return data_dict



