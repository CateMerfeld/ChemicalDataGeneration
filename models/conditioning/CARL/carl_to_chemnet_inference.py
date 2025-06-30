#%%
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
import importlib
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(parent_dir)
import functions as f
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data_preprocessing'))
sys.path.append(parent_dir)
import preprocessing_functions as pf
# Reload the functions module after updates
importlib.reload(f)

#%%
device = f.set_up_gpu()
save_file_path_pt_1 = '/home/cmdunham/ChemicalDataGeneration/data/encoder_embedding_predictions/carl_to_chemnet_conditional_' 
save_file_path_pt_2 = '_preds.csv'

train_file_path = '/home/cmdunham/scratch/CARL/train_carls_one_per_spec.feather'
val_file_path = '/home/cmdunham/scratch/CARL/val_carls_one_per_spec.feather'
test_file_path = '/home/cmdunham/scratch/CARL/test_carls_one_per_spec.feather'
encoder_path = '/home/cmdunham/ChemicalDataGeneration/models/trained_models/carl_to_chemnet_conditional_encoder.pth'
# best_model = f.Encoder().to(device)
# state_dict = torch.load(encoder_path, weights_only=False)
# best_model.load_state_dict(state_dict)
best_model = torch.load(encoder_path, map_location=device)
metadata = pd.read_feather('/home/cmdunham/scratch/BKG_SIM_ims_acbc_train_v1.1.09_meta.feather')


encoder_criterion = nn.MSELoss()
batch_size = 64
reparameterization = False
sorted_chem_names = ['DEB','DEM','DMMP','DPM','DtBP','JP8','MES','TEPO']
#%%
file_path = '/home/cmdunham/scratch/name_smiles_embedding_file.csv'
name_smiles_embedding_df = pd.read_csv(file_path)

# set the df index to be the chemical abbreviations in col 'Unnamed: 0'
name_smiles_embedding_df.set_index('Unnamed: 0', inplace=True)
name_smiles_embedding_df.head()

file_path = '/home/cmdunham/ChemicalDataGeneration/data/mass_spec_name_smiles_embedding_file.csv'
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
train_carls = pd.read_feather(train_file_path)
train_carls = train_carls.drop(columns=['level_0'])

train_embeddings_tensor, train_carl_tensor, train_chem_encodings_tensor, train_carl_indices_tensor = f.create_dataset_tensors(
    train_carls, 
    name_smiles_embedding_df, 
    device, 
    start_idx=1, 
    stop_idx=-9,
    )

# merge_conditions function expects a DataFrame with an 'index' column
train_embeddings_df = pd.DataFrame(train_embeddings_tensor.cpu().numpy())
train_embeddings_df['index'] = train_carl_indices_tensor.cpu().numpy()
# merge conditions with true embeddings
train_embeddings_with_conditions = pf.merge_conditions(train_embeddings_df, metadata, col_to_insert_before='index')
# drop the 'index' column after merging conditions since it's no longer needed
train_embeddings_with_conditions.drop(columns=['index'], inplace=True)
# replace NaN values with the mean of each column
train_embeddings_with_conditions = train_embeddings_with_conditions.fillna(train_embeddings_with_conditions.mean())
# scale the TemperatureKelvin column to be between 0 and 1
train_embeddings_with_conditions['TemperatureKelvin'] = (train_embeddings_with_conditions['TemperatureKelvin'] - train_embeddings_with_conditions['TemperatureKelvin'].min()) / (train_embeddings_with_conditions['TemperatureKelvin'].max() - train_embeddings_with_conditions['TemperatureKelvin'].min())
# scale the PressureBar column to be between 0 and 1
train_embeddings_with_conditions['PressureBar'] = (train_embeddings_with_conditions['PressureBar'] - train_embeddings_with_conditions['PressureBar'].min()) / (train_embeddings_with_conditions['PressureBar'].max() - train_embeddings_with_conditions['PressureBar'].min())
# convert the DataFrame back to a tensor
train_embeddings_tensor = torch.Tensor(train_embeddings_with_conditions.values).to(device)
del train_embeddings_with_conditions, train_carls

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

train_preds_file_path = save_file_path_pt_1 + 'train' + save_file_path_pt_2
train_preds_df.to_csv(train_preds_file_path, index=False)

del train_embeddings_tensor, train_carl_tensor, train_chem_encodings_tensor, train_carl_indices_tensor, train_preds_df
#%%

val_carls = pd.read_feather(val_file_path)
val_carls = val_carls.drop(columns=['level_0'])

val_embeddings_tensor, val_carl_tensor, val_chem_encodings_tensor, val_carl_indices_tensor = f.create_dataset_tensors(
    val_carls, 
    name_smiles_embedding_df, 
    device, 
    start_idx=1, 
    stop_idx=-9,
    )
# del val_carls
# merge_conditions function expects a DataFrame with an 'index' column
val_embeddings_df = pd.DataFrame(val_embeddings_tensor.cpu().numpy())
val_embeddings_df['index'] = val_carl_indices_tensor.cpu().numpy()
# merge conditions with true embeddings
val_embeddings_with_conditions = pf.merge_conditions(val_embeddings_df, metadata, col_to_insert_before='index')
# replace NaN values with the mean of each column
val_embeddings_with_conditions = val_embeddings_with_conditions.fillna(val_embeddings_with_conditions.mean())
# drop the 'index' column after merging conditions since it's no longer needed
val_embeddings_with_conditions.drop(columns=['index'], inplace=True)
# scale the TemperatureKelvin column to be between 0 and 1
val_embeddings_with_conditions['TemperatureKelvin'] = (val_embeddings_with_conditions['TemperatureKelvin'] - val_embeddings_with_conditions['TemperatureKelvin'].min()) / (val_embeddings_with_conditions['TemperatureKelvin'].max() - val_embeddings_with_conditions['TemperatureKelvin'].min())
# scale the PressureBar column to be between 0 and 1
val_embeddings_with_conditions['PressureBar'] = (val_embeddings_with_conditions['PressureBar'] - val_embeddings_with_conditions['PressureBar'].min()) / (val_embeddings_with_conditions['PressureBar'].max() - val_embeddings_with_conditions['PressureBar'].min())
# convert the DataFrame back to a tensor
val_embeddings_tensor = torch.Tensor(val_embeddings_with_conditions.values).to(device)
del val_embeddings_with_conditions, val_carls

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

val_preds_file_path = save_file_path_pt_1 + 'val' + save_file_path_pt_2
val_preds_df.to_csv(val_preds_file_path, index=False)

del val_embeddings_tensor, val_carl_tensor, val_chem_encodings_tensor, val_carl_indices_tensor, val_preds_df
#%%

test_carls = pd.read_feather(test_file_path)
test_carls = test_carls.drop(columns=['level_0'])
test_embeddings_tensor, test_carl_tensor, test_chem_encodings_tensor, test_carl_indices_tensor = f.create_dataset_tensors(
    test_carls, 
    name_smiles_embedding_df, 
    device, 
    start_idx=1, 
    stop_idx=-9,
    )
# del test_carls
# merge_conditions function expects a DataFrame with an 'index' column
test_embeddings_df = pd.DataFrame(test_embeddings_tensor.cpu().numpy())
test_embeddings_df['index'] = test_carl_indices_tensor.cpu().numpy()
# merge conditions with true embeddings
test_embeddings_with_conditions = pf.merge_conditions(test_embeddings_df, metadata, col_to_insert_before='index')
# drop the 'index' column after merging conditions since it's no longer needed
test_embeddings_with_conditions.drop(columns=['index'], inplace=True)
# replace NaN values with the mean of each column
test_embeddings_with_conditions = test_embeddings_with_conditions.fillna(test_embeddings_with_conditions.mean())
# scale the TemperatureKelvin column to be between 0 and 1
test_embeddings_with_conditions['TemperatureKelvin'] = (test_embeddings_with_conditions['TemperatureKelvin'] - test_embeddings_with_conditions['TemperatureKelvin'].min()) / (test_embeddings_with_conditions['TemperatureKelvin'].max() - test_embeddings_with_conditions['TemperatureKelvin'].min())
# scale the PressureBar column to be between 0 and 1
test_embeddings_with_conditions['PressureBar'] = (test_embeddings_with_conditions['PressureBar'] - test_embeddings_with_conditions['PressureBar'].min()) / (test_embeddings_with_conditions['PressureBar'].max() - test_embeddings_with_conditions['PressureBar'].min())
# convert the DataFrame back to a tensor
test_embeddings_tensor = torch.Tensor(test_embeddings_with_conditions.values).to(device)
del test_embeddings_with_conditions, test_carls

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

test_preds_file_path = save_file_path_pt_1 + 'test' + save_file_path_pt_2
test_preds_df.to_csv(test_preds_file_path, index=False)

del test_embeddings_tensor, test_carl_tensor, test_chem_encodings_tensor, test_carl_indices_tensor

#%%

# test_preds_df = pd.read_csv(test_preds_file_path)
# test_preds_df.head()
# sorted_chem_names = ['DEB','DEM','DMMP','DPM','DtBP','JP8','MES','TEPO']
# encodings_list = test_preds_df[sorted_chem_names].values.tolist()
# # spectra_labels = [sorted_chem_names[list(enc).index(1)] for enc in encodings_list]
# embeddings_only = test_preds_df.iloc[:,1:-8]
# embeddings_only.columns = ims_embeddings.T.columns
# # embeddings_only['Label'] = test['Label']
# embeddings_only['Label'] = [sorted_chem_names[enc.index(1)] for enc in encodings_list]
# pf.plot_emb_pca(
#     all_true_embeddings, embeddings_only, 'Test', 'IMS', 
#     log_wandb=False, chemnet_embeddings_to_plot=ims_embeddings,
#     show_wandb_run_name=False)