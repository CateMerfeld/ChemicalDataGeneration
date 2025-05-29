#%%
import pandas as pd
# import preprocessing_functions as pf
import matplotlib.pyplot as plt
#%%
metadata = pd.read_feather('../../scratch/BKG_SIM_ims_acbc_train_v1.1.09_meta.feather')
#%%
metadata.head()
#%%
temp = metadata['TemperatureKelvin']
print(f"Temperature range: {temp.min()} K to {temp.max()} K")

#%%
thing = metadata[metadata['TemperatureKelvin'] == 0]
print(f"Number of samples with zero temperature: {len(thing)}")
#%%
# sample_metadata = metadata.sample(n=1000, random_state=42)
plt.hist(metadata.loc[metadata['TemperatureKelvin'] != 0, 'TemperatureKelvin'], bins=30, edgecolor='black')
plt.xlabel('Temperature (Kelvin)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Histogram of TemperatureKelvin', fontsize=20)
plt.show()
#%%
plt.hist(metadata.loc[metadata['PressureBar'] != 0, 'PressureBar'], bins=30, edgecolor='black')
plt.xlabel('Pressure', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Histogram of PressureBar', fontsize=20)
plt.show()
#%%
plt.hist(metadata.loc[metadata['Label'] != 'BKG', 'PressureBar'], bins=30, edgecolor='black')
plt.xlabel('Pressure', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Histogram of PressureBar for Analytes', fontsize=20)
plt.show()


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


