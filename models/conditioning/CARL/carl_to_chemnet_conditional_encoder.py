#%%
import pandas as pd
# import numpy as np
from torch.utils.data import TensorDataset
import os
import importlib
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(parent_dir)
import functions as f
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data_preprocessing'))
sys.path.append(parent_dir)
import preprocessing_functions as pf
import torch
# # Reload the functions module after updates
# importlib.reload(f)

early_stopping_threshold = 15
model_hyperparams = {
  'batch_size':[32, 64],
  'epochs': [100],
  'learning_rate':[.001, .0001, .00001, .000001],
  }

# Loading Data:
metadata = pd.read_feather('/home/cmdunham/scratch/BKG_SIM_ims_acbc_train_v1.1.09_meta.feather')


file_path = '/home/cmdunham/scratch/CARL/train_carls_one_per_spec.feather'
train_carls = pd.read_feather(file_path)
train_carls = train_carls.drop(columns=['level_0'])
#%%

file_path = '/home/cmdunham/scratch/CARL/val_carls_one_per_spec.feather'
val_carls = pd.read_feather(file_path)
val_carls = val_carls.drop(columns=['level_0'])

file_path = '/home/cmdunham/scratch/CARL/test_carls_one_per_spec.feather'
test_carls = pd.read_feather(file_path)
test_carls = test_carls.drop(columns=['level_0'])

file_path = '/home/cmdunham/scratch/name_smiles_embedding_file.csv'
name_smiles_embedding_df = f.format_embedding_df(file_path) # pd.read_csv(file_path)


file_path = '/home/cmdunham/ChemicalDataGeneration/data/mass_spec_name_smiles_embedding_file.csv'
mass_spec_name_smiles_embedding_df = f.format_embedding_df(file_path) # pd.read_csv(file_path)

# Things that need to be changed for each encoder/dataset/target embedding
model_type = 'ConditionEncoder'
notebook_name = '/home/cmdunham/ChemicalDataGeneration/models/conditioning/CARL/carl_to_chemnet_conditional_encoder.py'
architecture = 'carl_conditional_encoder'
dataset_type = 'carls'
target_embedding = 'ChemNet'
encoder_path = '/home/cmdunham/ChemicalDataGeneration/models/trained_models/carl_to_chemnet_conditional_encoder.pth'

#%%
# Combine embeddings for IMS simulants and mass spec chems to use for plotting pca
ims_embeddings = pd.DataFrame([emb for emb in name_smiles_embedding_df['Embedding Floats']][1:]).T
mass_spec_embeddings = pd.DataFrame([emb for emb in mass_spec_name_smiles_embedding_df['Embedding Floats']]).T
cols = name_smiles_embedding_df.index[1:]
ims_embeddings.columns = cols
cols = mass_spec_name_smiles_embedding_df.index
mass_spec_embeddings.columns = cols
all_true_embeddings = pd.concat([ims_embeddings, mass_spec_embeddings], axis=1)
all_true_embeddings.head()
# #%%
# # mass_spec_name_smiles_embedding_df.head()
#%%
# Training Encoder on Carls:
device = f.set_up_gpu()

sorted_chem_names = list(train_carls.columns[-8:])
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
train_data_with_conditions = pf.merge_conditions(train_embeddings_df, metadata, col_to_insert_before='index')
# drop the 'index' column after merging conditions since it's no longer needed
train_data_with_conditions.drop(columns=['index'], inplace=True)
# replace NaN values with 0
train_data_with_conditions = train_data_with_conditions.fillna(0)
# convert the DataFrame back to a tensor
train_embeddings_tensor = torch.Tensor(train_data_with_conditions.values).to(device)
print(torch.isnan(train_embeddings_tensor).any())
del train_data_with_conditions, train_carls

val_embeddings_tensor, val_carl_tensor, val_chem_encodings_tensor, val_carl_indices_tensor = f.create_dataset_tensors(
    val_carls,     
    name_smiles_embedding_df, 
    device, 
    start_idx=1, 
    stop_idx=-9,
    )
# merge_conditions function expects a DataFrame with an 'index' column
val_embeddings_df = pd.DataFrame(val_embeddings_tensor.cpu().numpy())
val_embeddings_df['index'] = val_carl_indices_tensor.cpu().numpy()
# merge conditions with true embeddings
val_data_with_conditions = pf.merge_conditions(val_embeddings_df, metadata, col_to_insert_before='index')
# replace NaN values with 0
val_data_with_conditions = val_data_with_conditions.fillna(0)
# drop the 'index' column after merging conditions since it's no longer needed
val_data_with_conditions.drop(columns=['index'], inplace=True)

# convert the DataFrame back to a tensor
val_embeddings_tensor = torch.Tensor(val_data_with_conditions.values).to(device)

del val_data_with_conditions, val_carls


test_embeddings_tensor, test_carl_tensor, test_chem_encodings_tensor, test_carl_indices_tensor = f.create_dataset_tensors(
    test_carls, 
    name_smiles_embedding_df, 
    device, 
    start_idx=1, 
    stop_idx=-9,
    )
# merge_conditions function expects a DataFrame with an 'index' column
test_embeddings_df = pd.DataFrame(test_embeddings_tensor.cpu().numpy())
test_embeddings_df['index'] = test_carl_indices_tensor.cpu().numpy()
# merge conditions with true embeddings
test_data_with_conditions = pf.merge_conditions(test_embeddings_df, metadata, col_to_insert_before='index')
# drop the 'index' column after merging conditions since it's no longer needed
test_data_with_conditions.drop(columns=['index'], inplace=True)
# replace NaN values with 0
test_data_with_conditions = test_data_with_conditions.fillna(0)
# convert the DataFrame back to a tensor
test_embeddings_tensor = torch.Tensor(test_data_with_conditions.values).to(device)
del test_data_with_conditions, test_carls




config = {
    'wandb_entity': 'catemerfeld',
    'wandb_project': 'ims_encoder_decoder',
    'gpu':True,
    'threads':1,
}

os.environ['WANDB_NOTEBOOK_NAME'] = notebook_name

# Reload the functions module after updates
importlib.reload(f)

wandb_kwargs = {
    'architecture': architecture,
    'optimizer':'AdamW',
    'loss':'MSELoss',
    'dataset': dataset_type,
    'target_embedding': target_embedding,
    'early stopping threshold':early_stopping_threshold
}


train_data = TensorDataset(train_carl_tensor, train_chem_encodings_tensor, train_embeddings_tensor, train_carl_indices_tensor)
val_data = TensorDataset(val_carl_tensor, val_chem_encodings_tensor, val_embeddings_tensor, val_carl_indices_tensor)
test_data = TensorDataset(test_carl_tensor, test_chem_encodings_tensor, test_embeddings_tensor, test_carl_indices_tensor)

best_hyperparams = f.train_model(
    model_type, train_data, val_data, test_data, 
    device, config, wandb_kwargs, 
    all_true_embeddings, name_smiles_embedding_df, model_hyperparams, 
    sorted_chem_names, encoder_path, save_emb_pca_to_wandb=True, early_stop_threshold=early_stopping_threshold,
    input_type=dataset_type, show_wandb_run_name=True, lr_scheduler=True
    )