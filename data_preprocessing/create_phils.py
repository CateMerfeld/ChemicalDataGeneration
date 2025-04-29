#%%
import pandas as pd
#%%
import sys
import os
import preprocessing_functions as ppf
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models'))
sys.path.append(parent_dir)
import plotting_functions as pf
#%%
import importlib
#%%


#%%
importlib.reload(pf)
#%%
scaling_factor = .75
scaling_string = '75'
# %%
train_data_file = f'../../scratch/PHIL/train_phils_scaled_to_{scaling_string}_pct.csv'
train_data = pd.read_csv(train_data_file)
#%%
# train_data = ppf.load_data('../../scratch/train_data.feather')
train_data_file = f'../../scratch/PHIL/train_phils_scaled_to_{scaling_string}_pct.csv'
train_data = pd.read_feather(train_data_file)
#%%
sample_idx = 100

# sample = train_data.iloc[sample_idx:sample_idx+10,2:-9]
# sample = ppf.scale_reactant_ion_peak(sample, scaling_factor=scaling_factor)
# sample = sample.iloc[0]

sample = train_data.iloc[sample_idx,2:-9]
sample_label = train_data.iloc[sample_idx]['Label']
# save_plot_path = f'../plots/PHIL/{sample_label}_spectrum.png'
save_plot_path = f'../plots/PHIL/{sample_label}_scaled_to_{scaling_string}_pct_PHIL.png'
pf.plot_ims_spectrum(
    sample, sample_label, 'Experimental', 
    preprocessing_type = 'Spectrum', save_plot_path=save_plot_path
    )
#%%
import preprocessing_functions as ppf
print('Loading train data...')
train_data = ppf.load_data('../../scratch/train_data.feather')
train_data = ppf.scale_reactant_ion_peak(train_data, scaling_factor=scaling_factor)
train_data.to_feather(f'../../scratch/PHIL/train_phils_scaled_to_{scaling_string}_pct.csv')#, index=False)

del train_data
#%%

# print('Loading validation data...')
# val_data = ppf.load_data('../../scratch/val_data.feather')
# val_data = ppf.scale_reactant_ion_peak(val_data, scaling_factor=scaling_factor)
# # pf.plot_ims_spectrum(val_data.iloc[0,2:-9], val_data.iloc[0]['Label'], 'Experimental')
# val_data.to_feather(f'../../scratch/PHIL/val_phils_scaled_to_{scaling_string}_pct.csv')#, index=False)
# del val_data


# print('Loading test data...')
# test_data = ppf.load_data('../../scratch/test_data.feather')
# test_data = ppf.scale_reactant_ion_peak(test_data, scaling_factor=scaling_factor)
# test_data.to_feather(f'../../scratch/PHIL/test_phils_scaled_to_{scaling_string}_pct.csv')#, index=False)
# del test_data

