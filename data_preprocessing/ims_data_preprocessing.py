#%%
import pandas as pd
import preprocessing_functions as pf
#%%
metadata = pd.read_feather('../../scratch/BKG_SIM_ims_acbc_train_v1.1.09_meta.feather')

# Add temperature and pressure conditions to the data
#%%
data = pd.read_feather('../../scratch/train_data.feather')
data_with_conditions = pf.merge_conditions(data, metadata)
data_with_conditions.to_feather('../../scratch/train_data_with_conditions.feather')
del data_with_conditions, data
#%%
data = pd.read_feather('../../scratch/val_data.feather')
data_with_conditions = pf.merge_conditions(data, metadata)
data_with_conditions.to_feather('../../scratch/val_data_with_conditions.feather')
del data_with_conditions, data
#%%
data = pd.read_feather('../../scratch/test_data.feather')
data_with_conditions = pf.merge_conditions(data, metadata)
data_with_conditions.to_feather('../../scratch/test_data_with_conditions.feather')
del data_with_conditions, data


