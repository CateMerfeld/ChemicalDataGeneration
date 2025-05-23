#%%
import time
print('Loading libraries...')
start_time = time.time()
import pandas as pd
import sys
import os
import importlib
import sys
end_time = time.time()
print(f'Loaded in {round(end_time - start_time, 3)} seconds')
# %%
print('Loading plotting functions module...')
start_time = time.time()
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models'))
sys.path.append(parent_dir)
import plotting_functions as pf
end_time = time.time()
print(f'Loaded in {round(end_time - start_time, 3)} seconds')

# import numpy as np
# #%%
# print('Loading preprocessing functions module...')
# start_time = time.time()
# import preprocessing_functions as ppf
# end_time = time.time()
# print(f'Loaded in {round(end_time - start_time, 3)} seconds')

#%%
# file_path = '../data/train_test_val_splits/train_meta.csv'
# train_meta = pd.read_csv(file_path)
# file_path = '../../scratch/train_data.feather'
# train_spectra = pd.read_feather(file_path)

# #%%
# print(f"The range of values for 'TemperatureKelvin' is {train_meta['TemperatureKelvin'].min()} to {train_meta['TemperatureKelvin'].max()}")
# print((train_meta['TemperatureKelvin'].max() - train_meta['TemperatureKelvin'].min())/2 +train_meta['TemperatureKelvin'].min())
# print(f"The range of values for 'PressureBar' is {train_meta['PressureBar'].min()} to {train_meta['PressureBar'].max()}")
# print((train_meta['PressureBar'].max() - train_meta['PressureBar'].min())/2 +train_meta['PressureBar'].min())
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

# # #%%
# print('Creating condition dataframes...')
# ppf.load_data_save_condition_dfs(
#     '../data/train_test_val_splits/', '_meta.csv',
#     '../../scratch/', '_data.feather',
#     '../../scratch/', 'spectra', 
#     'PressureBar', condition_cutoff=1014
#     )

# print('Creating condition dataframes...')
# ppf.load_data_save_condition_dfs(
#     '../data/train_test_val_splits/', '_meta.csv',
#     '../../scratch/', '_data.feather',
#     '../../scratch/', 'spectra_cutoff_303', 
#     'TemperatureKelvin', condition_cutoff=303
#     )
# #%%
# importlib.reload(ppf)
# ppf.load_data_save_condition_dfs(
#     '../data/train_test_val_splits/', '_meta.csv',
#     '../data/carls/', '_carls_one_per_spec.feather',
#     '../data/train_test_val_splits/', 'carls', 
#     'TemperatureKelvin', condition_cutoff=302
#     )
# #%%
print('Loading low condition data...')
start_time = time.time()
# file_path = '../data/train_test_val_splits/train_carls_low_TemperatureKelvin.csv'
# file_path = '../data/train_test_val_splits/val_carls_low_TemperatureKelvin.csv'/
file_path = '../../scratch/train_spectra_low_TemperatureKelvin.csv'
# file_path = '../../scratch/val_spectra_low_PressureBar.csv'
# file_path = '/home/cmdunham/scratch/train_spectra_low_TemperatureKelvin.csv'
# file_path = '/home/cmdunham/scratch/val_spectra_cutoff_303_low_TemperatureKelvin.csv'
train_low_temp = pd.read_csv(file_path)
train_low_temp = train_low_temp.sample(frac=.1, random_state=42)
end_time = time.time()
print(f'Loaded in {round(end_time - start_time, 3)} seconds')
#%%
print('Loading high condition data...')
start_time = time.time()
# file_path = '../data/train_test_val_splits/train_carls_high_TemperatureKelvin.csv'
# file_path = '../data/train_test_val_splits/val_carls_high_TemperatureKelvin.csv'
# file_path = '../../scratch/val_spectra_high_PressureBar.csv'
file_path = '../../scratch/train_spectra_high_TemperatureKelvin.csv'
# file_path = '/home/cmdunham/scratch/train_spectra_high_TemperatureKelvin.csv'
# file_path = '/home/cmdunham/scratch/val_spectra_cutoff_303_high_TemperatureKelvin.csv'
train_high_temp = pd.read_csv(file_path)
train_high_temp = train_high_temp.sample(frac=.8, random_state=42)
end_time = time.time()
print(f'Loaded in {round(end_time - start_time, 3)} seconds')
#%%
# # # importlib.reload(pf)
condition = 'PressureBar'
save_file_path_pt1 = f'../plots/low_vs_high_{condition}_'
save_file_path_pt2 = '_spectra.png'
# def undersample_df(df, label_column):
#     min_count = df[label_column].value_counts().min()
#     return df.groupby(label_column).apply(lambda x: x.sample(min_count)).reset_index(drop=True)

# train_low_temp_sample = undersample_df(train_low_temp, 'Label')
# train_high_temp_sample = undersample_df(train_high_temp, 'Label')

importlib.reload(pf)
# for chem in train_high_temp['Label'].unique():
pf.plot_conditions_pca(
    train_low_temp.copy(), train_high_temp.copy(), 
    save_file_path_pt1, save_file_path_pt2, 
    condition, sample_size=5000
    )
#%%
print('Low Temp Label Counts:')
print(train_high_temp['Label'].value_counts())
print('-----------------------')
print('High Temp Label Counts:')
print(train_low_temp['Label'].value_counts())


# %%
# condition = 'PressureBar'
importlib.reload(pf)
condition = 'TemperatureKelvin'
avg_low_cond_spectra = []
avg_high_cond_spectra = []
for chem in train_high_temp['Label'].unique():
    high_temp_chem = train_high_temp[train_high_temp['Label'] == chem]
    low_temp_chem = train_low_temp[train_low_temp['Label'] == chem]
    avg_low_cond_spectrum, _, _ = pf.calculate_average_spectrum_and_percentiles(low_temp_chem.iloc[:,2:-9])
    avg_low_cond_spectra.append(avg_low_cond_spectrum)
    avg_high_cond_spectrum, _, _ = pf.calculate_average_spectrum_and_percentiles(high_temp_chem.iloc[:,2:-9])
    avg_high_cond_spectra.append(avg_high_cond_spectrum)
    # pf.plot_average_spectrum(
    #     low_temp_chem.iloc[:,2:-9], 
    #     high_temp_chem.iloc[:,2:-9], 
    #     chem, condition,
    #     save_file_path_pt1, save_file_path_pt2
    #     )
# avg_low_cond_spectra = pd.DataFrame(avg_low_cond_spectra)
# avg_low_cond_spectra['Label'] = train_high_temp['Label'].unique()
# avg_low_cond_spectra.to_csv(f'../data/average_low_temp_spectra.csv', index=False)
avg_high_cond_spectra = pd.DataFrame(avg_high_cond_spectra)
avg_high_cond_spectra['Label'] = train_high_temp['Label'].unique()
avg_high_cond_spectra.to_csv(f'../data/average_high_temp_spectra.csv', index=False)
#%%  

    
# print(avg_high_temp_spectrum.shape)
# #%%
# chem = 'MES'
# high_temp_chem = train_high_temp[train_high_temp['Label'] == chem]
# low_temp_chem = train_low_temp[train_low_temp['Label'] == chem]
# #%%
# import importlib
# importlib.reload(pf)
# #%%
# pf.plot_average_spectrum(
#     low_temp_chem.iloc[:,2:-9], 
#     high_temp_chem.iloc[:,2:-9], 
#     chem, 'TemperatureKelvin'
#     )