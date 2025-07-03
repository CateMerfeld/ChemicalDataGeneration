#%%
import pandas as pd
import numpy as np
#%%
from torch.utils.data import TensorDataset
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
#%%
import uninformative_embeddings_functions as adv_emb_f
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import functions as f
#%%

encoder_path = f'/home/cmdunham/ChemicalDataGeneration/trained_models/ims_to_onehot_encoder_9_layers.pth'
batch_size = 32
train_preds_file_path = '/home/cmdunham/ChemicalDataGeneration/data/encoder_embedding_predictions/uninformative_embeddings/ims_to_onehot_encoder_train_preds.feather'
val_preds_file_path = '/home/cmdunham/ChemicalDataGeneration/data/encoder_embedding_predictions/uninformative_embeddings/ims_to_onehot_encoder_val_preds.feather'
test_preds_file_path = '/home/cmdunham/ChemicalDataGeneration/data/encoder_embedding_predictions/uninformative_embeddings/ims_to_onehot_encoder_test_preds.feather'

# Things that need to be changed for each encoder/dataset/target embedding
notebook_name = '/home/cmdunham/ChemicalDataGeneration/models/uninformative_embeddings/ims_to_onehot_encoder.py'
architecture = 'ims_to_onehot_encoder'
dataset_type = 'ims'
target_embedding = 'OneHot'
model_type = 'IMStoOneHotEncoder'
# loss = 'CrossEntropyLoss'
loss = 'MSELoss'

early_stopping_threshold = 15
lr_scheduler_patience = 5
n_layers_list = [9]
model_hyperparams = {
  'batch_size':[32],
  'epochs': [100],
  'learning_rate':[.0001],
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
output_size = 512
#%%

# #%%
file_path = '/home/cmdunham/scratch/train_data.feather'
train_spectra = pd.read_feather(file_path)
sorted_chem_names = list(train_spectra.columns[-8:])

#%%
# format one-hot encoded embeddings correctly for f.create_dataset_tensors
# Create an 8x512 DataFrame: first 8 columns are one-hot, rest are zeros
onehot_matrix = np.zeros((8, 512), dtype=np.float32)
onehot_matrix[:, :8] = np.eye(8, dtype=np.float32)
cols = sorted_chem_names + [f"zero_{i}" for i in range(8, 512)]
embedding_df = pd.DataFrame(onehot_matrix, columns=cols, index=sorted_chem_names)
embedding_df['Embedding Floats'] = embedding_df.apply(lambda row: row.values.tolist(), axis=1)

#%%

class_weights = f.get_class_weights(train_spectra, device)
#%%

y_train, x_train, train_chem_encodings_tensor, train_indices_tensor = f.create_dataset_tensors(
    train_spectra, embedding_df, device, start_idx=start_idx, stop_idx=stop_idx)
del train_spectra

# #%%
file_path = '/home/cmdunham/scratch/val_data.feather'
val_spectra = pd.read_feather(file_path)
y_val, x_val, val_chem_encodings_tensor, val_indices_tensor = f.create_dataset_tensors(
    val_spectra, embedding_df, device, start_idx=start_idx, stop_idx=stop_idx)
del val_spectra
#%%
file_path = '/home/cmdunham/scratch/test_data.feather'
test_spectra = pd.read_feather(file_path)
y_test, x_test, test_chem_encodings_tensor, test_indices_tensor = f.create_dataset_tensors(
    test_spectra, embedding_df, device, start_idx=start_idx, stop_idx=stop_idx)
del test_spectra
#%%
mass_spec_name_smiles_embedding_df_file_path = '/home/cmdunham/ChemicalDataGeneration/data/mass_spec_name_smiles_embedding_file.csv'
# Combine ChemNet embeddings and OneHot embeddings so PCA plots are comparable
file_path = '/home/cmdunham/scratch/name_smiles_embedding_file.csv'
smiles_chemnet_embeddings_df = f.format_embedding_df(file_path)
chemnet_embeddings = pd.DataFrame([emb for emb in smiles_chemnet_embeddings_df['Embedding Floats']][1:]).T
cols = [f"{col} ChemNet" for col in smiles_chemnet_embeddings_df.index[1:]]
chemnet_embeddings.columns = cols
#%%

onehot_embeddings = pd.DataFrame([emb for emb in embedding_df['Embedding Floats']]).T
onehot_embeddings.columns = sorted_chem_names
all_true_embeddings = pd.concat([chemnet_embeddings, onehot_embeddings], axis=1)
#%%

# train_data = TensorDataset(x_train, train_chem_encodings_tensor, y_train, train_indices_tensor)
# val_data = TensorDataset(x_val, val_chem_encodings_tensor, y_val, val_indices_tensor)
# test_data = TensorDataset(x_test, test_chem_encodings_tensor, y_test, test_indices_tensor)

# for n_layers in n_layers_list:
#     wandb_kwargs['N Layers'] = n_layers
#     encoder_path = f'../trained_models/ims_to_onehot_encoder_{n_layers}_layers.pth'
#     base_model = adv_emb_f.IMStoOneHotEncoder(n_layers=n_layers, output_size=output_size)
#     if loss == 'CrossEntropyLoss':
#         criterion = nn.CrossEntropyLoss()#weight=class_weights)
#     elif loss == 'MSELoss':
#         criterion = nn.MSELoss()

#     best_hyperparams = adv_emb_f.train_model(
#         model_type, base_model, train_data, val_data, test_data, device, config, wandb_kwargs,
#         all_true_embeddings, onehot_embeddings, model_hyperparams, sorted_chem_names, 
#         encoder_path, criterion, input_type='IMS', embedding_type='OneHot',
#         early_stop_threshold=early_stopping_threshold, lr_scheduler=True, 
#         patience=lr_scheduler_patience, save_emb_pca_to_wandb=True
#         )
#%%
# batch_size = best_hyperparams['batch_size']

if loss == 'CrossEntropyLoss':
    criterion = nn.CrossEntropyLoss()
elif loss == 'MSELoss':
    criterion = nn.MSELoss()

mass_spec_name_smiles_embedding_df = f.format_embedding_df(mass_spec_name_smiles_embedding_df_file_path)
mass_spec_embeddings_df = pd.DataFrame([emb for emb in mass_spec_name_smiles_embedding_df['Embedding Floats']]).T #mass_spec_embeddings[chems_above_5]
cols = mass_spec_name_smiles_embedding_df.index
mass_spec_embeddings_df.columns = cols

ims_embeddings = pd.DataFrame([emb for emb in chemnet_embeddings['Embedding Floats']][1:]).T
cols = chemnet_embeddings.index[1:]
ims_embeddings.columns = cols
all_true_embeddings = pd.concat([ims_embeddings, mass_spec_embeddings_df], axis=1)

best_model = torch.load(encoder_path, weights_only=False)

print('Generating train embeddings...')
train_data = DataLoader(
    TensorDataset(
        x_train,
        train_chem_encodings_tensor, 
        y_train,
        train_indices_tensor
        ), 
    batch_size=batch_size, 
    shuffle=False
    )

predicted_embeddings, output_name_encodings, average_loss, input_indices = f.predict_embeddings(
    train_data, best_model, device, criterion)

train_preds_df = f.format_preds_df(
    input_indices, predicted_embeddings,
    output_name_encodings, sorted_chem_names
    )

train_preds_df.to_feather(train_preds_file_path)#, index=False)

# del train_embeddings_tensor, train_carl_tensor, train_chem_encodings_tensor, train_carl_indices_tensor, train_preds_df
# #%%
print('Generating validation embeddings...')
val_data = DataLoader(
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
    val_data, best_model, device, criterion)

val_preds_df = f.format_preds_df(
    input_indices, predicted_embeddings,
    output_name_encodings, sorted_chem_names
    )

val_preds_df.to_feather(val_preds_file_path)#, index=False)
#%%
print('Generating test embeddings...')

test_data = DataLoader(
    TensorDataset(
        x_test,
        test_chem_encodings_tensor, 
        y_test,
        test_indices_tensor
        ), 
    batch_size=batch_size, 
    shuffle=False
    )

predicted_embeddings, output_name_encodings, average_loss, input_indices = f.predict_embeddings(
    test_data, best_model, device, criterion)

test_preds_df = f.format_preds_df(
    input_indices, predicted_embeddings, 
    output_name_encodings, sorted_chem_names
    )

test_preds_df.to_feather(test_preds_file_path)#, index=False)
#%%