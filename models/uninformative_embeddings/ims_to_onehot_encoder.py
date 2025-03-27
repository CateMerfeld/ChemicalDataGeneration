#%%
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset
import os
import sys
import torch.nn as nn
#%%
import uninformative_embeddings_functions as adv_emb_f
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import functions as f

# Things that need to be changed for each encoder/dataset/target embedding
notebook_name = '/home/cmdunham/ChemicalDataGeneration/models/uninformative_embeddings/ims_to_onehot_encoder.py'
architecture = 'ims_to_onehot_encoder'
dataset_type = 'ims'
target_embedding = 'OneHot'
model_type = 'IMStoOneHotEncoder'
loss = 'CrossEntropyLoss'

early_stopping_threshold = 15
lr_scheduler_patience = 5
n_layers_list = [1,]#2,3,4]
model_hyperparams = {
  'batch_size':[32],
  'epochs': [10],
  'learning_rate':[.00001],
  }

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
# %%
device = f.set_up_gpu()
start_idx = 2
stop_idx = -9
#%%
# Loading Data:
file_path = '../../data/name_smiles_embedding_file.csv'
name_smiles_embedding_df = pd.read_csv(file_path)

# #%%
file_path = '../../../scratch/train_data.feather'
train_spectra = pd.read_feather(file_path)
sorted_chem_names = list(train_spectra.columns[-8:])

# format one-hot encoded embeddings correctly for f.create_dataset_tensors
true_embeddings = pd.get_dummies(sorted_chem_names, dtype=np.float32)
name_smiles_embedding_df = true_embeddings.copy()
name_smiles_embedding_df['Embedding Floats'] = true_embeddings.apply(lambda row: [row['DEB'], row['DEM'], row['DMMP'], row['DPM'], row['DtBP'], row['JP8'], row['MES'], row['TEPO']], axis=1)
name_smiles_embedding_df.set_index(name_smiles_embedding_df.columns[:-1], inplace=True)

class_weights = f.get_class_weights(train_spectra, device)

y_train, x_train, train_chem_encodings_tensor, train_carl_indices_tensor = f.create_dataset_tensors(
    train_spectra, name_smiles_embedding_df, device, start_idx=start_idx, stop_idx=stop_idx)
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
# %%

# Combine ChemNet embeddings and OneHot embeddings so PCA plots are comparable
file_path = '../../data/name_smiles_embedding_file.csv'
smiles_chemnet_embeddings_df = f.format_embedding_df(file_path)
chemnet_embeddings = pd.DataFrame([emb for emb in smiles_chemnet_embeddings_df['Embedding Floats']][1:]).T
# cols = smiles_chemnet_embeddings_df.index[1:]
cols = [f"{col} ChemNet" for col in smiles_chemnet_embeddings_df.index[1:]]
chemnet_embeddings.columns = cols

onehot_embeddings = pd.DataFrame([emb for emb in name_smiles_embedding_df['Embedding Floats']][1:]).T
cols = name_smiles_embedding_df.index[1:]
onehot_embeddings.columns = cols
zeros_df = pd.DataFrame(np.zeros((2, 7)), columns=onehot_embeddings.columns)
onehot_embeddings = pd.concat([zeros_df, onehot_embeddings], axis=0)
all_true_embeddings = pd.concat([chemnet_embeddings, onehot_embeddings], axis=1).reset_index(drop=True)
# all_true_embeddings.fillna(0, inplace=True)
#%%

train_data = TensorDataset(x_train, train_chem_encodings_tensor, y_train, train_carl_indices_tensor)
val_data = TensorDataset(x_val, val_chem_encodings_tensor, y_val, val_carl_indices_tensor)
test_data = TensorDataset(x_test, test_chem_encodings_tensor, y_test, test_carl_indices_tensor)

for n_layers in n_layers_list:
    wandb_kwargs['N Layers'] = n_layers
    encoder_path = f'../trained_models/ims_to_onehot_encoder_{n_layers}_layers.pth'
    base_model = adv_emb_f.Encoder(n_layers=n_layers)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_hyperparams = adv_emb_f.train_model(
        model_type, base_model, train_data, val_data, test_data, device, config, wandb_kwargs,
        all_true_embeddings, true_embeddings, model_hyperparams, sorted_chem_names, 
        encoder_path, criterion, input_type='IMS', embedding_type='OneHot',
        early_stop_threshold=early_stopping_threshold, lr_scheduler=True, patience=lr_scheduler_patience,
        )
# %%