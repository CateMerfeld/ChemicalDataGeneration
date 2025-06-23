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
# # Reload the functions module after updates
# importlib.reload(f)

early_stopping_threshold = 20
model_hyperparams = {
  'batch_size':[64],
  'epochs': [500],
  'learning_rate':[.0001],
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
notebook_name = '/home/cmdunham/ChemicalDataGeneration/models/carl_encoder.py'
architecture = 'carl_conditional_encoder'
dataset_type = 'carls'
target_embedding = 'ChemNet'
encoder_path = '/home/cmdunham/models/trained_models/carl_to_chemnet_conditional_encoder.pth'

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
#%%
mass_spec_name_smiles_embedding_df.head()
#%%
# Training Encoder on Carls:

device = f.set_up_gpu()
# train_embeddings_tensor, train_carl_tensor, train_chem_encodings_tensor, train_carl_indices_tensor = f.create_dataset_tensors(
#     train_carls, 
#     name_smiles_embedding_df, 
#     device, 
#     start_idx=1, 
#     stop_idx=-9,
#     )
val_embeddings_tensor, val_carl_tensor, val_chem_encodings_tensor, val_carl_indices_tensor = f.create_dataset_tensors(
    val_carls,     
    name_smiles_embedding_df, 
    device, 
    start_idx=1, 
    stop_idx=-9,
    )
print(val_carl_indices_tensor.shape, val_embeddings_tensor.shape)
# test_embeddings_tensor, test_carl_tensor, test_chem_encodings_tensor, test_carl_indices_tensor = f.create_dataset_tensors(
#     test_carls, 
#     name_smiles_embedding_df, 
#     device, 
#     start_idx=1, 
#     stop_idx=-9,
#     )
# sorted_chem_names = list(train_carls.columns[-8:])
# del train_carls, val_carls, test_carls



# config = {
#     'wandb_entity': 'catemerfeld',
#     'wandb_project': 'ims_encoder_decoder',
#     'gpu':True,
#     'threads':1,
# }

# os.environ['WANDB_NOTEBOOK_NAME'] = notebook_name

# # Reload the functions module after updates
# importlib.reload(f)

# wandb_kwargs = {
#     'architecture': architecture,
#     'optimizer':'AdamW',
#     'loss':'MSELoss',
#     'dataset': dataset_type,
#     'target_embedding': target_embedding,
#     'early stopping threshold':early_stopping_threshold
# }


# train_data = TensorDataset(train_carl_tensor, train_chem_encodings_tensor, train_embeddings_tensor, train_carl_indices_tensor)
# val_data = TensorDataset(val_carl_tensor, val_chem_encodings_tensor, val_embeddings_tensor, val_carl_indices_tensor)
# test_data = TensorDataset(test_carl_tensor, test_chem_encodings_tensor, test_embeddings_tensor, test_carl_indices_tensor)

# best_hyperparams = f.train_model(
#     'Encoder', train_data, val_data, test_data, 
#     device, config, wandb_kwargs, 
#     all_true_embeddings, name_smiles_embedding_df, model_hyperparams, 
#     sorted_chem_names, encoder_path, save_emb_pca_to_wandb=True, early_stop_threshold=early_stopping_threshold,
#     input_type='Carl', show_wandb_run_name=True, lr_scheduler=True
#     )