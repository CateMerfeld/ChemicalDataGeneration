#%%
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
import sys
#%%
import importlib
# from collections import Counter
# import importlib
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import plotting_functions as pf
import functions as f

#%%
device = f.set_up_gpu()
#%%
encoder_path = 'trained_models/ims_to_onehot_encoder.pth'
best_model = torch.load(encoder_path, weights_only=False)
encoder_criterion = nn.MSELoss()
batch_size = 16
#%%
sorted_chem_names = ['DEB','DEM','DMMP','DPM','DtBP','JP8','MES','TEPO']
all_true_embeddings = pd.get_dummies(sorted_chem_names, dtype=np.float32)
name_smiles_embedding_df = all_true_embeddings.copy()
name_smiles_embedding_df['Embedding Floats'] = all_true_embeddings.apply(lambda row: [row['DEB'], row['DEM'], row['DMMP'], row['DPM'], row['DtBP'], row['JP8'], row['MES'], row['TEPO']], axis=1)

name_smiles_embedding_df.set_index(name_smiles_embedding_df.columns[:-1], inplace=True)

# #%%
# file_path = '../../scratch/train_data.feather'
# train_spectra = pd.read_feather(file_path)
# #%%
# y_train, x_train, train_chem_encodings_tensor, train_indices_tensor = f.create_dataset_tensors(train_spectra, name_smiles_embedding_df, device)
# del train_spectra

# train_dataset = DataLoader(
#     TensorDataset(
#         x_train, 
#         train_chem_encodings_tensor, 
#         y_train,
#         train_indices_tensor
#         ), 
#         batch_size=batch_size, 
#         shuffle=False
#         )
# predicted_embeddings, output_name_encodings, average_loss, input_indices = f.predict_embeddings(
#     train_dataset, best_model, device, encoder_criterion)
# # input_carl_indices = [idx.cpu().detach().numpy() for idx_list in input_carl_indices for idx in idx_list]
# input_indices = [idx for idx_list in input_indices for idx in idx_list]
# # predicted_embeddings = [emb.cpu().detach().numpy() for emb_list in predicted_embeddings for emb in emb_list]
# predicted_embeddings = [emb for emb_list in predicted_embeddings for emb in emb_list]
# # output_name_encodings = [enc.cpu().detach().numpy() for enc_list in output_name_encodings for enc in enc_list]
# output_name_encodings = [enc for enc_list in output_name_encodings for enc in enc_list]
# train_preds_df = pd.DataFrame(predicted_embeddings)
# train_preds_df.insert(0, 'index', input_indices)
# name_encodings_df = pd.DataFrame(output_name_encodings)
# # name_encodings_df.columns = train_carls.columns[-8:]
# name_encodings_df.columns = sorted_chem_names
# train_preds_df = pd.concat([train_preds_df, name_encodings_df], axis=1)
# #%%
# file_path = '../data/encoder_embedding_predictions/ims_to_onehot_encoder_train_preds.csv'
# train_preds_df.to_csv(file_path, index=False)

# # del y_train, x_train, train_chem_encodings_tensor, train_carl_indices_tensor, train_preds_df

#%%
file_path = '../../scratch/val_data.feather'
val_spectra = pd.read_feather(file_path)

y_val, x_val, val_chem_encodings_tensor, val_indices_tensor = f.create_dataset_tensors(val_spectra, name_smiles_embedding_df, device)
del val_spectra

val_dataset = DataLoader(
    TensorDataset(
        x_val, 
        val_chem_encodings_tensor, 
        y_val,
        val_indices_tensor
        ), 
        batch_size=batch_size, 
        shuffle=False
        )
predicted_embeddings, output_name_encodings, average_loss, input_indices = f.predict_embeddings(
    val_dataset, best_model, device, encoder_criterion)
# input_carl_indices = [idx.cpu().detach().numpy() for idx_list in input_carl_indices for idx in idx_list]
input_indices = [idx for idx_list in input_indices for idx in idx_list]
# predicted_embeddings = [emb.cpu().detach().numpy() for emb_list in predicted_embeddings for emb in emb_list]
predicted_embeddings = [emb for emb_list in predicted_embeddings for emb in emb_list]
# output_name_encodings = [enc.cpu().detach().numpy() for enc_list in output_name_encodings for enc in enc_list]
output_name_encodings = [enc for enc_list in output_name_encodings for enc in enc_list]
val_preds_df = pd.DataFrame(predicted_embeddings)
val_preds_df.insert(0, 'index', input_indices)
name_encodings_df = pd.DataFrame(output_name_encodings)
# name_encodings_df.columns = train_carls.columns[-8:]
name_encodings_df.columns = sorted_chem_names
val_preds_df = pd.concat([val_preds_df, name_encodings_df], axis=1)

file_path = '../data/encoder_embedding_predictions/ims_to_onehot_encoder_val_preds.csv'
val_preds_df.to_csv(file_path, index=False)
#%%
# file_path = '../../scratch/test_data.feather'
# test_spectra = pd.read_feather(file_path)
# y_test, x_test, test_chem_encodings_tensor, test_indices_tensor = f.create_dataset_tensors(test_spectra, name_smiles_embedding_df, device)
# del test_spectra

# test_dataset = DataLoader(
#     TensorDataset(
#         x_test, 
#         test_chem_encodings_tensor, 
#         y_test,
#         test_indices_tensor
#         ), 
#         batch_size=batch_size, 
#         shuffle=False
#         )
# predicted_embeddings, output_name_encodings, average_loss, input_indices = f.predict_embeddings(
#     test_dataset, best_model, device, encoder_criterion)
# # input_carl_indices = [idx.cpu().detach().numpy() for idx_list in input_carl_indices for idx in idx_list]
# input_indices = [idx for idx_list in input_indices for idx in idx_list]
# # predicted_embeddings = [emb.cpu().detach().numpy() for emb_list in predicted_embeddings for emb in emb_list]
# predicted_embeddings = [emb for emb_list in predicted_embeddings for emb in emb_list]
# # output_name_encodings = [enc.cpu().detach().numpy() for enc_list in output_name_encodings for enc in enc_list]
# output_name_encodings = [enc for enc_list in output_name_encodings for enc in enc_list]
# test_preds_df = pd.DataFrame(predicted_embeddings)
# test_preds_df.insert(0, 'index', input_indices)
# name_encodings_df = pd.DataFrame(output_name_encodings)
# # name_encodings_df.columns = val_carls.columns[-8:]
# name_encodings_df.columns = sorted_chem_names
# test_preds_df = pd.concat([test_preds_df, name_encodings_df], axis=1)

# file_path = '../data/encoder_embedding_predictions/ims_to_onehot_encoder_test_preds.csv'
# test_preds_df.to_csv(file_path, index=False)

# # del y_test, x_test, test_chem_encodings_tensor, test_indices_tensor

# #%%
importlib.reload(pf)
#%%
file_path = '../../data/encoder_embedding_predictions/ims_to_onehot_encoder_test_preds.csv'
test_preds_df = pd.read_csv(file_path)
test_preds_df.head()
sorted_chem_names = ['DEB','DEM','DMMP','DPM','DtBP','JP8','MES','TEPO']
encodings_list = test_preds_df[sorted_chem_names].values.tolist()
# spectra_labels = [sorted_chem_names[list(enc).index(1)] for enc in encodings_list]
embeddings_only = test_preds_df.iloc[:,1:-8].copy()
embeddings_only.columns = all_true_embeddings.T.columns
# embeddings_only['Label'] = test['Label']

embeddings_only['Label'] = [sorted_chem_names[enc.index(1)] for enc in encodings_list]
pf.plot_emb_pca(
    all_true_embeddings, embeddings_only, 'Test', 'IMS', embedding_type='OneHot',
    log_wandb=False, chemnet_embeddings_to_plot=all_true_embeddings,
    show_wandb_run_name=False)