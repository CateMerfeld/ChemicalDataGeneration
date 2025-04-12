#%%
import pandas as pd
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
scaling_factor = .10
scaling_string = str(scaling_factor).split('.')[1]
print('Loading train data...')
train_data = ppf.load_data('../../scratch/train_data.feather')
train_data = ppf.scale_reactant_ion_peak(scaling_factor=scaling_factor)
train_data.to_csv(f'../../scratch/PHILs/train_phils_scaled_to_{scaling_string}_pct.csv', index=False)
del train_data


print('Loading validation data...')
val_data = ppf.load_data('../../scratch/val_data.feather')
val_data = ppf.scale_reactant_ion_peak(scaling_factor=scaling_factor)
# pf.plot_ims_spectrum(val_data.iloc[0,2:-9], val_data.iloc[0]['Label'], 'Experimental')
val_data.to_csv(f'../../scratch/PHILs/val_phils_scaled_to_{scaling_string}_pct.csv', index=False)
del val_data


print('Loading test data...')
test_data = ppf.load_data('../../scratch/test_data.feather')
test_data = ppf.scale_reactant_ion_peak(scaling_factor=scaling_factor)
test_data.to_csv(f'../../scratch/PHILs/test_phils_scaled_to_{scaling_string}_pct.csv', index=False)
del test_data

