#%%
import numpy as np
import pandas as pd
# import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import plotting_functions as pf
import functions as f
#%%
import importlib
importlib.reload(f)
#%%
file_path = '../../../scratch/train_data.feather'
train_spectra = pd.read_feather(file_path)
file_path = '../../data/encoder_embedding_predictions/ims_to_onehot_encoder_train_preds.csv'
train_embeddings = pd.read_csv(file_path)

# file_path = '../../../cmdunham/scratch/val_data.feather'
file_path = '../../../scratch/val_data.feather'
val_spectra = pd.read_feather(file_path)
# file_path = '/home/cmdunham/ChemicalDataGeneration/data/encoder_embedding_predictions/ims_to_onehot_encoder_val_preds.csv'
file_path = '../../data/encoder_embedding_predictions/ims_to_onehot_encoder_val_preds.csv'
val_embeddings = pd.read_csv(file_path)

file_path = '../../../scratch/test_data.feather'
test_spectra = pd.read_feather(file_path)
file_path = '../../data/encoder_embedding_predictions/ims_to_onehot_encoder_test_preds.csv'
test_embeddings = pd.read_csv(file_path)


#%%
device = f.set_up_gpu()
x_train, y_train, train_chem_encodings_tensor, train_indices_tensor = f.create_dataset_tensors_for_generator(train_spectra, train_embeddings, device, carl=False)
del train_spectra
train_data = TensorDataset(x_train, train_chem_encodings_tensor, y_train, train_indices_tensor)

x_val, y_val, val_chem_encodings_tensor, val_indices_tensor = f.create_dataset_tensors_for_generator(val_spectra, val_embeddings, device, carl=False)
del val_spectra
val_data = TensorDataset(x_val, val_chem_encodings_tensor, y_val, val_indices_tensor)

x_test, y_test, test_chem_encodings_tensor, test_indices_tensor = f.create_dataset_tensors_for_generator(test_spectra, test_embeddings, device, carl=False)
del test_spectra
test_data = TensorDataset(x_test, test_chem_encodings_tensor, y_test, test_indices_tensor)
#%%

sorted_chem_names = ['DEB','DEM','DMMP','DPM','DtBP','JP8','MES','TEPO']
config = {
    'wandb_entity': 'catemerfeld',
    'wandb_project': 'ims_encoder_decoder',
    'gpu':True,
    'threads':1,
}
notebook_name = '/home/cmdunham/ChemicalDataGeneration/models/onehot_to_ims_universal_generator.py'
os.environ['WANDB_NOTEBOOK_NAME'] = notebook_name

wandb_kwargs = {
    'architecture': 'universal_onehot_to_ims_generator',
    'optimizer':'AdamW',
    'loss':'MSELoss',
    'dataset': 'spectra',
    'target_embedding': 'OneHot',
    'early stopping threshold':20
}
model_hyperparams = {
    'batch_size':[16, 8],
    'epochs': [500],
    'learning_rate':[.01,.001, .0001],
    }

num_plots = 5
generator_path = '../trained_models/universal_onehot_to_ims_generator.pth'

f.train_generator(
    train_data, val_data, test_data, device, config, 
    wandb_kwargs, model_hyperparams, sorted_chem_names, 
    generator_path, early_stop_threshold=wandb_kwargs['early stopping threshold'], 
    lr_scheduler=True, num_plots=num_plots, plot_overlap_pca=True,
    model_type='OneHottoIMSGenerator'
    )