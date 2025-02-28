#%%
import pandas as pd
import sys
import os
import importlib
import sys
import preprocessing_functions as ppf
#%%
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models'))
sys.path.append(parent_dir)
import plotting_functions as pf
import numpy as np
#%%
# # import plotting_functions as pf
importlib.reload(ppf)
#%%
# file_path = '../data/train_test_val_splits/train_meta.csv'
# train_meta = pd.read_csv(file_path)
# file_path = '../../scratch/train_data.feather'
# train_spectra = pd.read_feather(file_path)

# #%%
# print(f"The range of values for 'TemperatureKelvin' is {train_meta['TemperatureKelvin'].min()} to {train_meta['TemperatureKelvin'].max()}")
# print((train_meta['TemperatureKelvin'].max() - train_meta['TemperatureKelvin'].min())/2 +train_meta['TemperatureKelvin'].min())
# print(f"The range of values for 'PressureBar' is {train_meta['PressureBar'].min()} to {train_meta['PressureBar'].max()}")
# #%%
# cutoff_temp = 303
# train_meta_low_temp = train_meta[train_meta['TemperatureKelvin'] < cutoff_temp]
# train_meta_high_temp = train_meta[train_meta['TemperatureKelvin'] >= cutoff_temp]

# print(f'There are {len(train_meta_low_temp)} samples with TemperatureKelvin < {cutoff_temp}')
# print(f'There are {len(train_meta_high_temp)} samples with TemperatureKelvin >= {cutoff_temp}')
# print('Low Temp Sample Counts:')
# print(train_meta_low_temp['Label'].value_counts())
# print('-----------------------')
# print('High Temp Sample Counts:')
# print(train_meta_high_temp['Label'].value_counts())

# chem = 'DPM'
# high_temp = train_meta_high_temp[train_meta_high_temp['Label'] == chem]
# low_temp = train_meta_low_temp[train_meta_low_temp['Label'] == chem]
# for _ in range(5):
#     i = np.random.randint(0, high_temp.shape[0])
#     high_temp_index = high_temp.iloc[i]['level_0']
#     # print(high_temp_index)
#     i = np.random.randint(0, low_temp.shape[0])
#     low_temp_index = low_temp.iloc[i]['level_0']
#     # print(low_temp_index)
#     high_temp_spectrum = train_spectra[train_spectra['index'] == high_temp_index].iloc[0, 2:-9]
#     low_temp_spectrum = train_spectra[train_spectra['index'] == low_temp_index].iloc[0, 2:-9]


#     pf.plot_spectra_real_synthetic_comparison(low_temp_spectrum, high_temp_spectrum, 'Spectrum', chem, left_plot_type='Low Temp', right_plot_type='High Temp')

# #%%
# ppf.load_data_save_condition_dfs(
#     '../data/train_test_val_splits/', '_meta.csv',
#     '../../scratch/', '_data.feather',
#     '../data/train_test_val_splits/', 'spectra', 'TemperatureKelvin'
#     )
# #%%
# importlib.reload(ppf)
# ppf.load_data_save_condition_dfs(
#     '../data/train_test_val_splits/', '_meta.csv',
#     '../data/carls/', '_carls_one_per_spec.feather',
#     '../data/train_test_val_splits/', 'carls', 
#     'TemperatureKelvin', condition_cutoff=302
#     )
#%%
# file_path = '../../data/train_test_val_splits/train_carls_low_TemperatureKelvin.csv'
file_path = '../data/train_test_val_splits/train_carls_low_TemperatureKelvin.csv'
train_carls_low_temp = pd.read_csv(file_path)
# file_path = '../../data/train_test_val_splits/train_carls_high_TemperatureKelvin.csv'
file_path = '../data/train_test_val_splits/train_carls_high_TemperatureKelvin.csv'
train_carls_high_temp = pd.read_csv(file_path)
# train_carls_combined = pd.concat([train_carls_low_temp, train_carls_high_temp], ignore_index=True)
#%%
importlib.reload(pf)
pf.plot_ims_spectra_pca(train_carls_high_temp, train_carls_low_temp)