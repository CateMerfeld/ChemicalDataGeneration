#%%
# Load Packages and Files:
import pandas as pd
#%%
import torch.nn as nn
# import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import plotting_functions as pf
import functions as f

# # # # Reload the functions module after updates
# # # import importlib
# # # importlib.reload(f)

#%%
start_idx = 2
stop_idx = -9

sample_frac = .1

device = f.set_up_gpu()
encoder_type = ''
model_type = 'universal_generator_limited_training_data'
model_target = 'High Temp Spectra'
# chem_group = ['DMMP', 'TEPO']
## %%

# file_path = '../../data/train_test_val_splits/train_carls_high_TemperatureKelvin.csv'
file_path = '../../../scratch/train_spectra_high_TemperatureKelvin.csv'
train_data = pd.read_csv(file_path).sample(frac=sample_frac, random_state=42)
file_path = f'../../data/encoder_embedding_predictions/conditioning_train_preds{encoder_type}.csv'
train_embeddings_df = pd.read_csv(file_path)
# train_embeddings_df += np.random.normal(0, 0.01, train_embeddings_df.shape)

# # train generator on subset of chemicals
# train_data = train_data[train_data['Label'].isin(chem_group)].copy()

## There are MORE low temp preds than high temp carls. Shuffle/repeat the high temp carls to match the number of low temp preds
train_data, train_embeddings_df = f.oversample_condition_data(train_data, train_embeddings_df)

x_train, y_train, train_chem_encodings_tensor, train_carl_indices_tensor = f.create_dataset_tensors_for_generator(
    train_data, train_embeddings_df, device, start_idx=start_idx, stop_idx=stop_idx)

del train_data, train_embeddings_df
#%%
# file_path = '../../data/train_test_val_splits/val_carls_high_TemperatureKelvin.csv'
file_path = '../../../scratch/val_spectra_high_TemperatureKelvin.csv'
val_data = pd.read_csv(file_path).sample(frac=sample_frac, random_state=42)
file_path = f'../../data/encoder_embedding_predictions/conditioning_val_preds{encoder_type}.csv'
val_embeddings_df = pd.read_csv(file_path)
# val_embeddings_df += np.random.normal(0, 0.01, val_embeddings_df.shape)

# val_data = val_data[val_data['Label'].isin(chem_group)].copy()
val_data, val_embeddings_df = f.oversample_condition_data(val_data, val_embeddings_df)

x_val, y_val, val_chem_encodings_tensor, val_carl_indices_tensor = f.create_dataset_tensors_for_generator(
    val_data, val_embeddings_df, device, start_idx=start_idx, stop_idx=stop_idx)

del val_data, val_embeddings_df
# # %%
# file_path = '../../data/train_test_val_splits/test_carls_high_TemperatureKelvin.csv'/
file_path = '../../../scratch/test_spectra_high_TemperatureKelvin.csv'
test_data = pd.read_csv(file_path).sample(frac=sample_frac, random_state=42)
file_path = f'../../data/encoder_embedding_predictions/conditioning_test_preds{encoder_type}.csv'
test_embeddings_df = pd.read_csv(file_path)
# test_embeddings_df += np.random.normal(0, 0.01, test_embeddings_df.shape)

# test_data = test_data[test_data['Label'].isin(chem_group)].copy()
test_data, test_embeddings_df = f.oversample_condition_data(test_data, test_embeddings_df)

x_test, y_test, test_chem_encodings_tensor, test_carl_indices_tensor = f.create_dataset_tensors_for_generator(
    test_data, test_embeddings_df, device, start_idx=start_idx, stop_idx=stop_idx)
test_cols = test_data.columns[2:-9]

del test_data, test_embeddings_df
# %%

sorted_chem_names = ['DEB','DEM','DMMP','DPM','DtBP','JP8','MES','TEPO']

train_data = TensorDataset(x_train, train_chem_encodings_tensor, y_train, train_carl_indices_tensor)
val_data = TensorDataset(x_val, val_chem_encodings_tensor, y_val, val_carl_indices_tensor)
test_data = TensorDataset(x_test, test_chem_encodings_tensor, y_test, test_carl_indices_tensor)

#%%
wandb_kwargs = {
    'architecture': model_type,
    'optimizer':'AdamW',
    'loss':'MSELoss',
    'input': 'Low Temp Embeddings',
    'target': model_target,
    'early stopping threshold': 15
}
model_hyperparams = {
    'batch_size':[16],
    'epochs': [500],
    'learning_rate':[.01],
    # 'freeze_layers': [8]
    }

config = {
    'wandb_entity': 'catemerfeld',
    'wandb_project': 'ims_encoder_decoder',
    'gpu':True,
    'threads':1,
}
notebook_name = '/home/cmdunham/ChemicalDataGeneration/models/conditional_generator.py'
num_plots = 5
# generator_load_path = '../trained_models/low_temp_embedding_to_high_temp_carl_conditional_generator.pth'
# generator_save_path = f'../trained_models/low_temp_embedding_to_high_temp_spectrum_conditional_{group}_group_generator.pth'
# generator_save_path = f'../trained_models/low_temp_embedding_to_high_temp_spectrum_conditional_universal_generator_undertrained.pth'
generator_save_path = f'../trained_models/low_temp_embedding_to_high_temp_spectrum_conditional_universal_generator{model_type}.pth'
carl_or_spec = 'Spectrum'

f.train_generator(
    train_data, val_data, test_data, device, config,
    wandb_kwargs, model_hyperparams, sorted_chem_names,
    generator_save_path, save_plots_to_wandb=False,
    early_stop_threshold=wandb_kwargs['early stopping threshold'], 
    num_plots=num_plots, # pretrained_model_path=generator_load_path,
    carl_or_spec=carl_or_spec
)
# # #%%
# # # #%%
# # Get predictions and save synthetic spectra
# batch_size = 16
# freeze_layers = 8
# criterion = nn.MSELoss()
# # group = 'dmmp_tepo'
# group = 'all_chemicals'
# model_type = 'universal_generator_undertrained'

# device = f.set_up_gpu()
# # generator_path = f'../trained_models/low_temp_embedding_to_high_temp_spectrum_conditional_{group}_group_generator.pth'
# # generator_path = '../trained_models/low_temp_embedding_to_high_temp_carl_conditional_generator.pth'
# generator_path = generator_save_path
# condition = 'high_TemperatureKelvin'
# preds_save_path = f'../../data/ims_data/synthetic_test_{condition}_{group}_{model_type}_spectra.csv'

# model = f.load_model(generator_path,device=device, freeze_layers=freeze_layers)

# test_dataset = DataLoader(test_data, batch_size)
# test_preds, test_chem_name_encodings, _, test_indices = f.predict_embeddings(test_dataset, model, device, criterion)

# test_chem_name_encodings_list = [enc for enc_list in test_chem_name_encodings for enc in enc_list]
# test_labels = [sorted_chem_names[list(enc).index(1)] for enc in test_chem_name_encodings_list]

# synthetic_spectra = [pred for pred_list in test_preds for pred in pred_list]
# synthetic_spectra_df = pd.DataFrame(synthetic_spectra, columns=test_cols)
# synthetic_spectra_df['Label'] = test_labels

# synthetic_spectra_df.to_csv(preds_save_path, index=False)