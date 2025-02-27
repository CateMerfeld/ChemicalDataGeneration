#%%
import pandas as pd
import sys
import os
import importlib
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models'))
sys.path.append(parent_dir)
import plotting_functions as pf
#%%
# import plotting_functions as pf
importlib.reload(pf)
file_path = '../data/train_test_val_splits/train_meta.csv'
train_meta = pd.read_csv(file_path)
file_path = '../../scratch/train_data.feather'
train_spectra = pd.read_feather(file_path)

#%%
print(f"The range of values for 'TemperatureKelvin' is {train_meta['TemperatureKelvin'].min()} to {train_meta['TemperatureKelvin'].max()}")
print((train_meta['TemperatureKelvin'].max() - train_meta['TemperatureKelvin'].min())/2 +train_meta['TemperatureKelvin'].min())
print(f"The range of values for 'PressureBar' is {train_meta['PressureBar'].min()} to {train_meta['PressureBar'].max()}")
#%%
cutoff_temp = 303
train_meta_low_temp = train_meta[train_meta['TemperatureKelvin'] < cutoff_temp]
train_meta_high_temp = train_meta[train_meta['TemperatureKelvin'] >= cutoff_temp]

print(f'There are {len(train_meta_low_temp)} samples with TemperatureKelvin < {cutoff_temp}')
print(f'There are {len(train_meta_high_temp)} samples with TemperatureKelvin >= {cutoff_temp}')
print('Low Temp Sample Counts:')
print(train_meta_low_temp['Label'].value_counts())
print('-----------------------')
print('High Temp Sample Counts:')
print(train_meta_high_temp['Label'].value_counts())
#%%

chem = 'DMMP'
high_temp = train_meta_high_temp[train_meta_high_temp['Label'] == chem]
low_temp = train_meta_low_temp[train_meta_low_temp['Label'] == chem]
for i in range(5):
    high_temp_index = high_temp.iloc[i]['level_0']
    low_temp_index = low_temp.iloc[i]['level_0']
    # Get the carls data for the high and low temp samples
    high_temp_spectra = train_spectra[train_spectra['index'] == high_temp_index].iloc[0, 2:-9]
    low_temp_spectra = train_spectra[train_spectra['index'] == low_temp_index].iloc[0, 2:-9]

    pf.plot_spectra_real_synthetic_comparison(low_temp_spectra, high_temp_spectra, 'Spectrum', chem, left_plot_type='Low Temp', right_plot_type='High Temp')