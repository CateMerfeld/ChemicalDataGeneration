#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import wandb
import os
from sklearn.decomposition import PCA
import itertools

from collections import Counter
import importlib
import functions as f
# Reload the functions module after updates
importlib.reload(f)

#%%
device = f.set_up_gpu()
#%%
encoder_path = 'trained_models/carl_to_chemnet_encoder_reparameterization.pth'
best_model = torch.load(encoder_path, weights_only=False)
encoder_criterion = nn.MSELoss()
batch_size = 16
reparameterization = True
sorted_chem_names = ['DEB','DEM','DMMP','DPM','DtBP','JP8','MES','TEPO']
#%%
file_path = '../data/name_smiles_embedding_file.csv'
name_smiles_embedding_df = pd.read_csv(file_path)

# set the df index to be the chemical abbreviations in col 'Unnamed: 0'
name_smiles_embedding_df.set_index('Unnamed: 0', inplace=True)
name_smiles_embedding_df.head()

file_path = '../data/mass_spec_name_smiles_embedding_file.csv'
mass_spec_name_smiles_embedding_df = pd.read_csv(file_path)

# set the df index to be the chemical abbreviations in col 'Unnamed: 0'
mass_spec_name_smiles_embedding_df.set_index('Unnamed: 0', inplace=True)
mass_spec_name_smiles_embedding_df.head()
embedding_floats = []
for chem_name in name_smiles_embedding_df.index:
    if chem_name == 'BKG':
        embedding_floats.append(None)
    else:
        embedding_float = name_smiles_embedding_df['embedding'][chem_name].split('[')[1]
        embedding_float = embedding_float.split(']')[0]
        embedding_float = [np.float32(num) for num in embedding_float.split(',')]
        embedding_floats.append(embedding_float)

name_smiles_embedding_df['Embedding Floats'] = embedding_floats
mass_spec_embedding_floats = []
for chem_name in mass_spec_name_smiles_embedding_df.index:
    embedding_float = mass_spec_name_smiles_embedding_df['embedding'][chem_name].split('[')[1]
    embedding_float = embedding_float.split(']')[0]
    embedding_float = [np.float32(num) for num in embedding_float.split(',')]
    mass_spec_embedding_floats.append(embedding_float)

mass_spec_name_smiles_embedding_df['Embedding Floats'] = mass_spec_embedding_floats
filtered_mass_spec_embeddings = pd.DataFrame([emb for emb in mass_spec_name_smiles_embedding_df['Embedding Floats']]).T #mass_spec_embeddings[chems_above_5]
cols = mass_spec_name_smiles_embedding_df.index
filtered_mass_spec_embeddings.columns = cols

# Combine embeddings for IMS simulants and mass spec chems to use for plotting pca
ims_embeddings = pd.DataFrame([emb for emb in name_smiles_embedding_df['Embedding Floats']][1:]).T
cols = name_smiles_embedding_df.index[1:]
ims_embeddings.columns = cols
all_true_embeddings = pd.concat([ims_embeddings, filtered_mass_spec_embeddings], axis=1)
all_true_embeddings.head()


#%%
file_path = '../data/carls/train_carls_one_per_spec.feather'
train_carls = pd.read_feather(file_path)
train_carls = train_carls.drop(columns=['level_0'])

train_embeddings_tensor, train_carl_tensor, train_chem_encodings_tensor, train_carl_indices_tensor = f.create_dataset_tensors(train_carls, name_smiles_embedding_df, device, carl=True)
del train_carls
train_dataset = DataLoader(
    TensorDataset(
        train_carl_tensor, 
        train_chem_encodings_tensor, 
        train_embeddings_tensor,
        train_carl_indices_tensor
        ), 
        batch_size=batch_size, 
        shuffle=False
        )
predicted_embeddings, output_name_encodings, average_loss, input_carl_indices = f.predict_embeddings(
    train_dataset, best_model, device, encoder_criterion, reparameterization)
# input_carl_indices = [idx.cpu().detach().numpy() for idx_list in input_carl_indices for idx in idx_list]
input_carl_indices = [idx for idx_list in input_carl_indices for idx in idx_list]
# predicted_embeddings = [emb.cpu().detach().numpy() for emb_list in predicted_embeddings for emb in emb_list]
predicted_embeddings = [emb for emb_list in predicted_embeddings for emb in emb_list]
# output_name_encodings = [enc.cpu().detach().numpy() for enc_list in output_name_encodings for enc in enc_list]
output_name_encodings = [enc for enc_list in output_name_encodings for enc in enc_list]
train_preds_df = pd.DataFrame(predicted_embeddings)
train_preds_df.insert(0, 'index', input_carl_indices)
name_encodings_df = pd.DataFrame(output_name_encodings)
# name_encodings_df.columns = train_carls.columns[-8:]
name_encodings_df.columns = sorted_chem_names
train_preds_df = pd.concat([train_preds_df, name_encodings_df], axis=1)

file_path = '../data/encoder_embedding_predictions/reparameterization_train_preds.csv'
train_preds_df.to_csv(file_path, index=False)

del train_embeddings_tensor, train_carl_tensor, train_chem_encodings_tensor, train_carl_indices_tensor, train_preds_df
#%%
file_path = '../data/carls/val_carls_one_per_spec.feather'
val_carls = pd.read_feather(file_path)
val_carls = val_carls.drop(columns=['level_0'])

val_embeddings_tensor, val_carl_tensor, val_chem_encodings_tensor, val_carl_indices_tensor = f.create_dataset_tensors(val_carls, name_smiles_embedding_df, device, carl=True)
del val_carls

val_dataset = DataLoader(
    TensorDataset(
        val_carl_tensor, 
        val_chem_encodings_tensor, 
        val_embeddings_tensor,
        val_carl_indices_tensor
        ), 
        batch_size=batch_size, 
        shuffle=False
        )
predicted_embeddings, output_name_encodings, average_loss, input_carl_indices = f.predict_embeddings(
    val_dataset, best_model, device, encoder_criterion, reparameterization)
# input_carl_indices = [idx.cpu().detach().numpy() for idx_list in input_carl_indices for idx in idx_list]
input_carl_indices = [idx for idx_list in input_carl_indices for idx in idx_list]
# predicted_embeddings = [emb.cpu().detach().numpy() for emb_list in predicted_embeddings for emb in emb_list]
predicted_embeddings = [emb for emb_list in predicted_embeddings for emb in emb_list]
# output_name_encodings = [enc.cpu().detach().numpy() for enc_list in output_name_encodings for enc in enc_list]
output_name_encodings = [enc for enc_list in output_name_encodings for enc in enc_list]
val_preds_df = pd.DataFrame(predicted_embeddings)
val_preds_df.insert(0, 'index', input_carl_indices)
name_encodings_df = pd.DataFrame(output_name_encodings)
# name_encodings_df.columns = val_carls.columns[-8:]
name_encodings_df.columns = sorted_chem_names
val_preds_df = pd.concat([val_preds_df, name_encodings_df], axis=1)

file_path = '../data/encoder_embedding_predictions/reparameterization_val_preds.csv'
val_preds_df.to_csv(file_path, index=False)

del val_embeddings_tensor, val_carl_tensor, val_chem_encodings_tensor, val_carl_indices_tensor, val_preds_df
#%%
file_path = '../data/carls/test_carls_one_per_spec.feather'
test_carls = pd.read_feather(file_path)
test_carls = test_carls.drop(columns=['level_0'])
test_embeddings_tensor, test_carl_tensor, test_chem_encodings_tensor, test_carl_indices_tensor = f.create_dataset_tensors(test_carls, name_smiles_embedding_df, device, carl=True)
del test_carls

test_dataset = DataLoader(
    TensorDataset(
        test_carl_tensor, 
        test_chem_encodings_tensor, 
        test_embeddings_tensor,
        test_carl_indices_tensor
        ), 
        batch_size=batch_size, 
        shuffle=False
        )
predicted_embeddings, output_name_encodings, average_loss, input_carl_indices = f.predict_embeddings(
    test_dataset, best_model, device, encoder_criterion, reparameterization)
# input_carl_indices = [idx.cpu().detach().numpy() for idx_list in input_carl_indices for idx in idx_list]
input_carl_indices = [idx for idx_list in input_carl_indices for idx in idx_list]
# predicted_embeddings = [emb.cpu().detach().numpy() for emb_list in predicted_embeddings for emb in emb_list]
predicted_embeddings = [emb for emb_list in predicted_embeddings for emb in emb_list]
# output_name_encodings = [enc.cpu().detach().numpy() for enc_list in output_name_encodings for enc in enc_list]
output_name_encodings = [enc for enc_list in output_name_encodings for enc in enc_list]
test_preds_df = pd.DataFrame(predicted_embeddings)
test_preds_df.insert(0, 'index', input_carl_indices)
name_encodings_df = pd.DataFrame(output_name_encodings)
# name_encodings_df.columns = val_carls.columns[-8:]
name_encodings_df.columns = sorted_chem_names
test_preds_df = pd.concat([test_preds_df, name_encodings_df], axis=1)

file_path = '../data/encoder_embedding_predictions/reparameterization_test_preds.csv'
test_preds_df.to_csv(file_path, index=False)

del test_embeddings_tensor, test_carl_tensor, test_chem_encodings_tensor, test_carl_indices_tensor

#%%

file_path = '../data/encoder_embedding_predictions/reparameterization_test_preds.csv'
test_preds_df = pd.read_csv(file_path)
test_preds_df.head()
sorted_chem_names = ['DEB','DEM','DMMP','DPM','DtBP','JP8','MES','TEPO']
encodings_list = test_preds_df[sorted_chem_names].values.tolist()
# spectra_labels = [sorted_chem_names[list(enc).index(1)] for enc in encodings_list]
embeddings_only = test_preds_df.iloc[:,1:-8]
embeddings_only.columns = ims_embeddings.T.columns
# embeddings_only['Label'] = test['Label']
embeddings_only['Label'] = [sorted_chem_names[enc.index(1)] for enc in encodings_list]
f.plot_emb_pca(
    all_true_embeddings, embeddings_only, 'Test', 'IMS', 
    log_wandb=False, chemnet_embeddings_to_plot=ims_embeddings,
    show_wandb_run_name=False)