# General script to run encoder model
import pandas as pd
from torch.utils.data import TensorDataset
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
# import plotting_functions as pf
import functions as f

#%%
# Things that need to be changed for each encoder/dataset/target embedding
notebook_name = '/home/cmdunham/ChemicalDataGeneration/models/run_encoder.py'
architecture = 'ims_to_ChemNet_encoder'
dataset_type = 'ims'
target_embedding = 'ChemNet'
# encoder_path = f'../trained_models/ims_to_chemnet_encoder_{n_layers}_layers.pth'
model_type = 'Encoder'
loss = 'MSELoss'

name_smiles_embedding_df_file_path = '../../data/name_smiles_embedding_file.csv'
train_file_path = '../../../scratch/train_data.feather'
val_file_path = '../../../scratch/val_data.feather'
test_file_path = '../../../scratch/test_data.feather'

early_stopping_threshold = 15
lr_scheduler_patience = 5
# n_layers_list = [2,3,4]

model_hyperparams = {
  'batch_size':[32],
  'epochs': [100],
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

device = f.set_up_gpu()
start_idx = 2
stop_idx = -9
#%%
# Loading Data:
name_smiles_embedding_df = f.format_embedding_df(name_smiles_embedding_df_file_path)

embeddings_df = pd.DataFrame([emb for emb in name_smiles_embedding_df['Embedding Floats']][1:]).T
cols = name_smiles_embedding_df.index[1:]
embeddings_df.columns = cols
#%%
train_data = pd.read_feather(train_file_path)

y_train, x_train, train_chem_encodings_tensor, train_indices_tensor = f.create_dataset_tensors(
    train_data, name_smiles_embedding_df, device, start_idx=start_idx, stop_idx=stop_idx)
sorted_chem_names = list(train_data.columns[-8:])
del train_data

# %%
val_data = pd.read_feather(val_file_path)
y_val, x_val, val_chem_encodings_tensor, val_indices_tensor = f.create_dataset_tensors(
    val_data, name_smiles_embedding_df, device, start_idx=start_idx, stop_idx=stop_idx)
del val_data
#%%
test_data = pd.read_feather(test_file_path)
y_test, x_test, test_chem_encodings_tensor, test_indices_tensor = f.create_dataset_tensors(
    test_data, name_smiles_embedding_df, device, start_idx=start_idx, stop_idx=stop_idx)
del test_data
#%%

train_data = TensorDataset(x_train, train_chem_encodings_tensor, y_train, train_indices_tensor)
val_data = TensorDataset(x_val, val_chem_encodings_tensor, y_val, val_indices_tensor)
test_data = TensorDataset(x_test, test_chem_encodings_tensor, y_test, test_indices_tensor)


# # Not sure if this part needs to be changed, nothing in here has been checked
# for n_layers in n_layers_list:
#     wandb_kwargs['N Layers'] = n_layers
#     encoder_path = f'../trained_models/ims_to_chemnet_encoder_{n_layers}_layers.pth'
#     base_model = adv_emb_f.Encoder(n_layers=n_layers, output_size=512)
#     criterion = nn.MSELoss()

#     best_hyperparams = adv_emb_f.train_model(
#         model_type, base_model, train_data, val_data, test_data, device, config, wandb_kwargs,
#         embeddings_df , name_smiles_embedding_df, model_hyperparams, sorted_chem_names, 
#         encoder_path, criterion, input_type='IMS', embedding_type='OneHot',
#         early_stop_threshold=early_stopping_threshold, lr_scheduler=True, patience=lr_scheduler_patience,
#         )