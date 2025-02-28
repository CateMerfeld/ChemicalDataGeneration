#%%
# Load Packages and Files:
import pandas as pd
#%%
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import os
import importlib
import functions as f
import sys
import importlib
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import plotting_functions as pf
import functions as f
# # Reload the functions module after updates
# importlib.reload(f)
#%%
start_idx = 2
stop_idx = -9

device = f.set_up_gpu()
#%%
file_path = '../../data/train_test_val_splits/train_carls_low_TemperatureKelvin.csv'
train_carls = pd.read_csv(file_path)
file_path = '../../data/encoder_embedding_predictions/conditioning_train_preds.csv'
train_preds_df = pd.read_csv(file_path)

x_train, y_train, train_chem_encodings_tensor, train_carl_indices_tensor = f.create_dataset_tensors_for_generator(train_carls, train_preds_df, device)

del train_carls
#%%
file_path = '../../data/train_test_val_splits/val_carls_low_TemperatureKelvin.csv'
val_carls = pd.read_csv(file_path)
file_path = '../../data/encoder_embedding_predictions/conditioning_val_preds.csv'
val_preds_df = pd.read_csv(file_path)

x_val, y_val, val_chem_encodings_tensor, val_carl_indices_tensor = f.create_dataset_tensors_for_generator(val_carls, val_preds_df, device)

del val_carls
#%%
file_path = '../../data/train_test_val_splits/test_carls_low_TemperatureKelvin.csv'
test_carls = pd.read_csv(file_path)
file_path = '../../data/encoder_embedding_predictions/conditioning_test_preds.csv'
test_preds_df = pd.read_csv(file_path)

x_test, y_test, test_chem_encodings_tensor, test_carl_indices_tensor = f.create_dataset_tensors_for_generator(test_carls, test_preds_df, device)

del test_carls
#%%

# sorted_chem_names = ['DEB','DEM','DMMP','DPM','DtBP','JP8','MES','TEPO']

# file_path_dict = {'train_carls_file_path':'../../data/train_test_val_splits/train_carls_high_TemperatureKelvin.csv',
#                 'train_embeddings_file_path':'../../data/encoder_embedding_predictions/conditioning_train_preds.csv',

#                 'val_carls_file_path': '../../data/carls/val_carls_one_per_spec.feather', 
#                 'val_embeddings_file_path': '../../data/encoder_embedding_predictions/conditioning_val_preds.csv',

#                 'test_carls_file_path':'../../data/carls/test_carls_one_per_spec.feather', 
#                 'test_embeddings_file_path':  '../../data/encoder_embedding_predictions/conditioning_test_preds.csv',
#                 }
# #%%
# wandb_kwargs = {
#     'architecture': 'conditional_universal_carl_generator',
#     'optimizer':'AdamW',
#     'loss':'MSELoss',
#     'dataset': 'high_temp_carls',
#     'target_embedding': 'ChemNet',
#     'early stopping threshold':20
# }
# model_hyperparams = {
#     'batch_size':[4,16],
#     'epochs': [500],
#     'learning_rate':[.001,.0001],
#     }
# notebook_name = '/home/cmdunham/ChemicalDataGeneration/models/individual_generator.py'
# num_plots = 5
# chem_to_run='MES'

# for chem in sorted_chem_names:
#     # generator_path= f'../models/trained_models/{chem}_carl_to_chemnet_generator_reparameterization.pth'
#     generator_path = f'../models/trained_models/{chem}_carl_to_chemnet_generator.pth'
#     if chem == chem_to_run:
#         f.run_generator(file_path_dict, chem, model_hyperparams, wandb_kwargs, sorted_chem_names, generator_path, notebook_name, num_plots)
