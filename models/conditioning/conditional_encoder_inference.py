#%%
import pandas as pd
# import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import importlib
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import plotting_functions as pf
import functions as f
#%%
importlib.reload(pf)
#%%

device = f.set_up_gpu()
#%%
encoder_path = '../trained_models/low_temp_carl_to_chemnet_encoder.pth'
best_model = torch.load(encoder_path, weights_only=False)
encoder_criterion = nn.MSELoss()
batch_size = 16
sorted_chem_names = ['DEB','DEM','DMMP','DPM','DtBP','JP8','MES','TEPO']
#%%
file_path = '../../data/name_smiles_embedding_file.csv'
name_smiles_embedding_df = f.format_embedding_df(file_path)
file_path = '../../data/mass_spec_name_smiles_embedding_file.csv'
mass_spec_name_smiles_embedding_df = f.format_embedding_df(file_path)

all_true_embeddings = f.combine_embeddings(name_smiles_embedding_df, mass_spec_name_smiles_embedding_df)

#%%
start_idx = 2
stop_idx = -9

#%%
file_path = '../../data/train_test_val_splits/train_carls_low_TemperatureKelvin.csv'
train_carls = pd.read_csv(file_path)
y_train, x_train, train_chem_encodings_tensor, train_carl_indices_tensor = f.create_dataset_tensors(
    train_carls, name_smiles_embedding_df, device, start_idx=start_idx, stop_idx=stop_idx)
sorted_chem_names = list(train_carls.columns[-8:])
del train_carls

train_dataset = DataLoader(
    TensorDataset(
        x_train, train_chem_encodings_tensor, y_train,
        train_carl_indices_tensor
        ), 
        batch_size=batch_size, shuffle=False
        )
predicted_embeddings, output_name_encodings, average_loss, input_carl_indices = f.predict_embeddings(
    train_dataset, best_model, device, encoder_criterion)
train_preds_df = f.format_preds_df(input_carl_indices, predicted_embeddings, output_name_encodings, sorted_chem_names)

file_path = '../../data/encoder_embedding_predictions/conditioning_train_preds.csv'
train_preds_df.to_csv(file_path, index=False)

del x_train, y_train, train_chem_encodings_tensor, train_carl_indices_tensor
#%%
file_path = '../../data/train_test_val_splits/val_carls_low_TemperatureKelvin.csv'
val_carls = pd.read_csv(file_path)
y_val, x_val, val_chem_encodings_tensor, val_carl_indices_tensor = f.create_dataset_tensors(
    val_carls, name_smiles_embedding_df, device, start_idx=start_idx, stop_idx=stop_idx)
sorted_chem_names = list(val_carls.columns[-8:])
del val_carls

val_dataset = DataLoader(
    TensorDataset(
        x_val, val_chem_encodings_tensor, y_val,
        val_carl_indices_tensor
        ), 
        batch_size=batch_size, shuffle=False
        )
predicted_embeddings, output_name_encodings, average_loss, input_carl_indices = f.predict_embeddings(
    val_dataset, best_model, device, encoder_criterion)
val_preds_df = f.format_preds_df(input_carl_indices, predicted_embeddings, output_name_encodings, sorted_chem_names)

file_path = '../../data/encoder_embedding_predictions/conditioning_val_preds.csv'
val_preds_df.to_csv(file_path, index=False)

del x_val, y_val, val_chem_encodings_tensor, val_carl_indices_tensor
#%%
file_path = '../../data/train_test_val_splits/test_carls_low_TemperatureKelvin.csv'
test_carls = pd.read_csv(file_path)
y_test, x_test, test_chem_encodings_tensor, test_carl_indices_tensor = f.create_dataset_tensors(
    test_carls, name_smiles_embedding_df, device, start_idx=start_idx, stop_idx=stop_idx)
sorted_chem_names = list(test_carls.columns[-8:])
del test_carls

test_dataset = DataLoader(
    TensorDataset(
        x_test, test_chem_encodings_tensor, y_test,
        test_carl_indices_tensor
        ), 
        batch_size=batch_size, shuffle=False
        )
predicted_embeddings, output_name_encodings, average_loss, input_carl_indices = f.predict_embeddings(
    test_dataset, best_model, device, encoder_criterion)
test_preds_df = f.format_preds_df(input_carl_indices, predicted_embeddings, output_name_encodings, sorted_chem_names)

file_path = '../../data/encoder_embedding_predictions/conditioning_test_preds.csv'
test_preds_df.to_csv(file_path, index=False)

del x_test, y_test, test_chem_encodings_tensor, test_carl_indices_tensor

#%%
file_path = '../../data/encoder_embedding_predictions/conditioning_test_preds.csv'
test_preds_df = pd.read_csv(file_path)
ims_embeddings = pd.DataFrame([emb for emb in name_smiles_embedding_df['Embedding Floats']][1:]).T
cols = name_smiles_embedding_df.index[1:]
ims_embeddings.columns = cols

sorted_chem_names = ['DEB','DEM','DMMP','DPM','DtBP','JP8','MES','TEPO']
encodings_list = test_preds_df[sorted_chem_names].values.tolist()
embeddings_only = test_preds_df.iloc[:,1:-8].copy()
embeddings_only.columns = ims_embeddings.T.columns
embeddings_only['Label'] = [sorted_chem_names[enc.index(1)] for enc in encodings_list]

pf.plot_emb_pca(
    all_true_embeddings, embeddings_only, 'Test Low Temp', 'IMS', 
    log_wandb=False, chemnet_embeddings_to_plot=ims_embeddings,
    show_wandb_run_name=False)