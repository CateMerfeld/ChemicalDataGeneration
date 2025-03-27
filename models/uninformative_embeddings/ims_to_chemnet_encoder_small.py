#%%
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset
import os
import sys
import torch.nn as nn
#%%
import uninformative_embeddings_functions as u_e_f
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import plotting_functions as pf
import functions as f
#%%
# Things that need to be changed for each encoder/dataset/target embedding
# repeating the 4 layer model to check if performance is the same or if 
# model is being reused (performance improves)
n_layers_list = [4,4]#,6,8]
notebook_name = '/home/cmdunham/ChemicalDataGeneration/models/uninformative_embeddings/ims_to_chemnet_small.py'
architecture = 'ims_to_ChemNet_encoder_small'
dataset_type = 'ims'
target_embedding = 'ChemNet'
# encoder_path = f'../trained_models/ims_to_chemnet_encoder_{n_layers}_layers.pth'
model_type = 'Encoder'
loss = 'CrossEntropyLoss'
early_stopping_threshold = 15
lr_scheduler_patience = 5

model_hyperparams = {
  'batch_size':[32],
  'epochs': [2],
  'learning_rate':[.00001],
  }

device = f.set_up_gpu()
start_idx = 2
stop_idx = -9
#%%
# Loading Data:
file_path = '../../data/name_smiles_embedding_file.csv'
name_smiles_embedding_df = f.format_embedding_df(file_path)

#%%
file_path = '../../../scratch/train_data.feather'
train_spectra = pd.read_feather(file_path)

class_weights = f.get_class_weights(train_spectra, device)

y_train, x_train, train_chem_encodings_tensor, train_carl_indices_tensor = f.create_dataset_tensors(
    train_spectra, name_smiles_embedding_df, device, start_idx=start_idx, stop_idx=stop_idx)
sorted_chem_names = list(train_spectra.columns[-8:])
del train_spectra

#%%
file_path = '../../../scratch/val_data.feather'
val_spectra = pd.read_feather(file_path)
y_val, x_val, val_chem_encodings_tensor, val_carl_indices_tensor = f.create_dataset_tensors(
    val_spectra, name_smiles_embedding_df, device, start_idx=start_idx, stop_idx=stop_idx)
del val_spectra
#%%
file_path = '../../../scratch/test_data.feather'
test_spectra = pd.read_feather(file_path)
y_test, x_test, test_chem_encodings_tensor, test_carl_indices_tensor = f.create_dataset_tensors(
    test_spectra, name_smiles_embedding_df, device, start_idx=start_idx, stop_idx=stop_idx)
del test_spectra



#%%

config = {
    'wandb_entity': 'catemerfeld',
    'wandb_project': 'ims_encoder_decoder',
    'gpu':True,
    'threads':1,
}

os.environ['WANDB_NOTEBOOK_NAME'] = notebook_name

wandb_kwargs = {
    'architecture': architecture,
    'optimizer':'AdamW',
    'loss': loss,
    'dataset': dataset_type,
    'target_embedding': target_embedding,
    'early stopping threshold':early_stopping_threshold
}

train_data = TensorDataset(x_train, train_chem_encodings_tensor, y_train, train_carl_indices_tensor)
val_data = TensorDataset(x_val, val_chem_encodings_tensor, y_val, val_carl_indices_tensor)
test_data = TensorDataset(x_test, test_chem_encodings_tensor, y_test, test_carl_indices_tensor)

for n_layers in n_layers_list:
    encoder_path = f'../trained_models/ims_to_chemnet_encoder_{n_layers}_layers.pth'
    base_model = u_e_f.Encoder(n_layers=n_layers, output_size=512)
    criterion = nn.MSELoss()

    best_hyperparams = u_e_f.train_model(
        model_type, base_model, train_data, val_data, test_data, device, config, wandb_kwargs,
        name_smiles_embedding_df , name_smiles_embedding_df, model_hyperparams, sorted_chem_names, 
        encoder_path, criterion, input_type='IMS', embedding_type='OneHot',
        early_stop_threshold=early_stopping_threshold, lr_scheduler=True, patience=lr_scheduler_patience,
        )
