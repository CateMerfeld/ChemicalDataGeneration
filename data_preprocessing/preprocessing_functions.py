import pandas as pd
# import requests

# from fcd_torch import FCD
# import torch

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder

# import numpy as np
# import time
# import GPUtil

# import matplotlib.pyplot as plt

def load_data(file_path):
    if file_path.endswith('.feather'):
        data = pd.read_feather(file_path)
    elif file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a .feather or .csv file.")
    return data

def scale_reactant_ion_peak(data, scaling_factor=.1, rip_start_col='184', rip_stop_col='300'):
    """
    Scale reactant ion peak by a given factor.
    
    Parameters:
    scaling_factor (float): Factor by which to scale the reactant ion peak.
    rip_start_col (int): Start column name for the reactant ion peak.
    rip_stop_col (int): Stop column name for the reactant ion peak.
    
    Returns:
    pd.DataFrame: DataFrame containing the scaled data.
    """

    for spec_type in ['p_', 'n_']:
        rip_start_idx = data.columns.get_loc(f'{spec_type}{rip_start_col}')
        rip_stop_idx = data.columns.get_loc(f'{spec_type}{rip_stop_col}')
        # Multiply all intensity values between rip_start_idx and rip_stop_idx by scaling_factor
        data.iloc[:, rip_start_idx:rip_stop_idx] *= scaling_factor
    
    return data

def create_condition_dfs(metadata, spectra, condition, condition_cutoff):
    low_condition_meta = metadata[metadata[condition] < condition_cutoff]
    low_condition_indices = low_condition_meta['level_0']
    low_condition_spectra = spectra[spectra['index'].isin(low_condition_indices)]
    high_condition_meta = metadata[metadata[condition] >= condition_cutoff]
    high_condition_indices = high_condition_meta['level_0']
    high_condition_spectra = spectra[spectra['index'].isin(high_condition_indices)]
    return low_condition_meta, low_condition_spectra, high_condition_meta, high_condition_spectra

def load_data_save_condition_dfs(
    meta_file_pt1, meta_file_pt2, spectra_file_pt1, spectra_file_pt2, 
    save_file_pt1, save_file_pt2, condition, condition_cutoff
    ):
    for split_type in ['train', 'val', 'test']:
        file_path = meta_file_pt1 + split_type + meta_file_pt2
        split_meta = pd.read_csv(file_path)
        file_path = spectra_file_pt1 + split_type + spectra_file_pt2
        split_spectra = pd.read_feather(file_path)
        # print(split_type.upper(), 'value counts:')
        # print(split_spectra['Label'].value_counts())
        _, split_spectra_low_condition, _, split_spectra_high_condition = create_condition_dfs(split_meta, split_spectra, condition, condition_cutoff)
        split_spectra_high_condition.to_csv(save_file_pt1 + split_type + '_' + save_file_pt2 + '_high_' + condition + '.csv', index=False)
        # print(split_type.upper(), f'low {condition} value counts:')
        # print(split_spectra_low_condition['Label'].value_counts())
        # print(split_type.upper(), f'high {condition} value counts:')
        # print(split_spectra_high_condition['Label'].value_counts())
        # print('---------------------------------')
        # print('---------------------------------')
        split_spectra_low_condition.to_csv(save_file_pt1 + split_type + '_' + save_file_pt2 + '_low_' + condition + '.csv', index=False)
        del split_spectra, split_meta, split_spectra_high_condition, split_spectra_low_condition