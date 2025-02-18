#%%
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset
import os
import importlib
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import plotting_functions as pf
import functions as f
# Reload the functions module after updates
# importlib.reload(f)

# Loading Data:
file_path = '../../../scratch/train_data.feather'
train_spectra = pd.read_feather(file_path)
file_path = '../../../scratch/val_data.feather'
val_spectra = pd.read_feather(file_path)
file_path = '../../../scratch/test_data.feather'
test_spectra = pd.read_feather(file_path)

# file_path = '../data/name_smiles_embedding_file.csv'
# name_smiles_embedding_df = pd.read_csv(file_path)
#%%
# format one-hot encoded embeddings correctly for f.create_dataset_tensors
sorted_chem_names = list(train_spectra.columns[-8:])
# sorted_chem_names = list(val_spectra.columns[-8:])
true_embeddings = pd.get_dummies(sorted_chem_names, dtype=np.float32)
# print(true_embeddings.head())
#%%
# from sklearn.decomposition import PCA
# pca = PCA(n_components=2)
# true_embeddings_pca = pca.fit_transform(true_embeddings.T)
#%%
name_smiles_embedding_df = true_embeddings.copy()
name_smiles_embedding_df['Embedding Floats'] = true_embeddings.apply(lambda row: [row['DEB'], row['DEM'], row['DMMP'], row['DPM'], row['DtBP'], row['JP8'], row['MES'], row['TEPO']], axis=1)

name_smiles_embedding_df.set_index(name_smiles_embedding_df.columns[:-1], inplace=True)

#%%

# Training Encoder on One-Hot Encoded labels:
device = f.set_up_gpu()
y_train, x_train, train_chem_encodings_tensor, train_carl_indices_tensor = f.create_dataset_tensors(train_spectra, name_smiles_embedding_df, device)
del train_spectra
y_val, x_val, val_chem_encodings_tensor, val_carl_indices_tensor = f.create_dataset_tensors(val_spectra, name_smiles_embedding_df, device)
del val_spectra
y_test, x_test, test_chem_encodings_tensor, test_carl_indices_tensor = f.create_dataset_tensors(test_spectra, name_smiles_embedding_df, device)
del test_spectra
#%%
# Things that need to be changed for each encoder/dataset/target embedding
notebook_name = '/home/cmdunham/ChemicalDataGeneration/models/uninformative_embeddings/ims_to_onehot_encoder.py'
architecture = 'ims_to_onehot_encoder'
dataset_type = 'ims'
target_embedding = 'OneHot'
encoder_path = '../trained_models/ims_to_onehot_encoder.pth'
model_type = 'IMStoOneHotEncoder'

config = {
    'wandb_entity': 'catemerfeld',
    'wandb_project': 'ims_encoder_decoder',
    'gpu':True,
    'threads':1,
}

os.environ['WANDB_NOTEBOOK_NAME'] = notebook_name

# Reload the functions module after updates
importlib.reload(f)
early_stopping_threshold = 15
wandb_kwargs = {
    'architecture': architecture,
    'optimizer':'AdamW',
    'loss':'MSELoss',
    'dataset': dataset_type,
    'target_embedding': target_embedding,
    'early stopping threshold':early_stopping_threshold
}

model_hyperparams = {
  'batch_size':[8,4],
  'epochs': [500],
  'learning_rate':[.00001, .000001],
  }

train_data = TensorDataset(x_train, train_chem_encodings_tensor, y_train, train_carl_indices_tensor)
val_data = TensorDataset(x_val, val_chem_encodings_tensor, y_val, val_carl_indices_tensor)
test_data = TensorDataset(x_test, test_chem_encodings_tensor, y_test, test_carl_indices_tensor)

best_hyperparams = f.train_model(
    model_type, train_data, val_data, test_data, 
    device, config, wandb_kwargs, true_embeddings, 
    true_embeddings, model_hyperparams, sorted_chem_names, 
    encoder_path, save_emb_pca_to_wandb=True, early_stop_threshold=early_stopping_threshold,
    input_type=dataset_type, show_wandb_run_name=True, lr_scheduler=True
    )