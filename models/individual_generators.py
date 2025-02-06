#%%
# Load Packages and Files:
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset 
import wandb
import os
from collections import Counter
import importlib
import functions as f
import GPUtil

#%%
# Reload the functions module after updates
importlib.reload(f)
#%%
## Loading ChemNet Embeddings:
file_path = '../data/name_smiles_embedding_file.csv'
name_smiles_embedding_df = pd.read_csv(file_path)

# set the df index to be the chemical abbreviations in col 'Unnamed: 0'
name_smiles_embedding_df.set_index('Unnamed: 0', inplace=True)
name_smiles_embedding_df.head()
file_path = '../data/all_chemnet_embeddings.csv'
all_true_embeddings = pd.read_csv(file_path)
#%%

## Loading CARLS and CARL-Based Embedding Preds:
chem = 'DEM'
device = f.set_up_gpu()

#%%
file_path = '../data/carls/train_carls_one_per_spec.feather'
train_carls = pd.read_feather(file_path)
train_carls.drop('level_0', axis=1, inplace=True)

file_path = '../data/encoder_embedding_predictions/train_embeddings.csv'
train_embeddings = pd.read_csv(file_path)

train_embeddings_tensor, train_carl_tensor, train_chem_encodings_tensor, train_carl_indices_tensor = f.create_individual_chemical_dataset_tensors(train_carls, train_embeddings, device, chem)
# del train_carls, train_embeddings
# #%%

# file_path = '../data/carls/val_carls_one_per_spec.feather'
# val_carls = pd.read_feather(file_path)
# val_carls.drop('level_0', axis=1, inplace=True)

# file_path = '../data/encoder_embedding_predictions/val_embeddings.csv'
# val_embeddings = pd.to_csv(file_path)

# val_embeddings_tensor, val_carl_tensor, val_chem_encodings_tensor, val_carl_indices_tensor = f.create_individual_chemical_dataset_tensors(val_carls, val_embeddings, device, chem)
# # del val_carls, val_embeddings

# #%%
# file_path = '../data/carls/test_carls_one_per_spec.feather'
# test_carls = pd.read_feather(file_path)
# test_carls.drop('level_0', axis=1, inplace=True)

# file_path = '../data/encoder_embedding_predictions/test_embeddings.csv'
# test_embeddings = pd.to_csv(file_path)

# test_embeddings_tensor, test_carl_tensor, test_chem_encodings_tensor, test_carl_indices_tensor = f.create_individual_chemical_dataset_tensors(test_carls, test_embeddings, device, chem)
# # del test_carls, test_embeddings
# #%%

# # Training Generator:

# # device = f.set_up_gpu()
# # chem = 'DEM'
# # train_embeddings_tensor, train_carl_tensor, train_chem_encodings_tensor, train_carl_indices_tensor = f.create_individual_chemical_dataset_tensors(train_carls, train_embeddings, device, chem)
# # val_embeddings_tensor, val_carl_tensor, val_chem_encodings_tensor, val_carl_indices_tensor = f.create_individual_chemical_dataset_tensors(val_carls, val_embeddings, device, chem)
# # test_embeddings_tensor, test_carl_tensor, test_chem_encodings_tensor, test_carl_indices_tensor = f.create_individual_chemical_dataset_tensors(test_carls, test_embeddings, device, chem)
# #%%
# # Things that need to be changed for each generator/dataset/target carl
# notebook_name = '/home/cmdunham/ChemicalDataGeneration/models/individual_ims_generator.ipynb'
# architecture = 'individual_carl_generator'
# dataset_type = 'carls'
# target_embedding = 'ChemNet'
# generator_path = f'../models/trained_models/{chem}_carl_to_chemnet_generator.pth'

# config = {
#     'wandb_entity': 'catemerfeld',
#     'wandb_project': 'ims_encoder_decoder',
#     'gpu':True,
#     'threads':1,
# }

# os.environ['WANDB_NOTEBOOK_NAME'] = notebook_name

# #%%
# wandb_kwargs = {
#     'architecture': architecture,
#     'optimizer':'AdamW',
#     'loss':'MSELoss',
#     'dataset': dataset_type,
#     'target_embedding': target_embedding
# }

# sorted_chem_names = list(val_carls.columns[-8:])

# # model_hyperparams = {
# #   'batch_size':[32, 64],
# #   'epochs': [500],
# #   'learning_rate':[.001, .0001],
# #   }

# model_hyperparams = {
#   'batch_size':[64],
#   'epochs': [500],
#   'learning_rate':[.01],
#   }

# train_data = TensorDataset(train_embeddings_tensor, train_chem_encodings_tensor, train_carl_tensor, train_carl_indices_tensor)
# val_data = TensorDataset(val_embeddings_tensor, val_chem_encodings_tensor, val_carl_tensor, val_carl_indices_tensor)
# test_data = TensorDataset(test_embeddings_tensor, test_chem_encodings_tensor, test_carl_tensor, test_carl_indices_tensor)

# f.train_generator(
#     train_data, val_data, test_data, device, config, 
#     wandb_kwargs, model_hyperparams, sorted_chem_names, 
#     generator_path, early_stop_threshold=15, 
#     lr_scheduler=True, num_plots=5
#     )