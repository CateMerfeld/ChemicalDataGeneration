#%%
import pandas as pd
# import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
import sys
import importlib

#%%
importlib.reload(pf)
#%%
# from collections import Counter
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import plotting_functions as pf
import functions as f

#%%
device = f.set_up_gpu()
#%%
file_path = '../../data/name_smiles_embedding_file.csv'
name_smiles_embedding_df = f.format_embedding_df(file_path)

ims_embeddings = pd.DataFrame([emb for emb in name_smiles_embedding_df['Embedding Floats']][1:]).T
cols = name_smiles_embedding_df.index[1:]
ims_embeddings.columns = cols

file_path = '../../data/mass_spec_name_smiles_embedding_file.csv'
mass_spec_name_smiles_embedding_df = f.format_embedding_df(file_path)
mass_spec_embeddings = pd.DataFrame([emb for emb in mass_spec_name_smiles_embedding_df['Embedding Floats']]).T #mass_spec_embeddings[chems_above_5]
cols = mass_spec_name_smiles_embedding_df.index
mass_spec_embeddings.columns = cols

all_true_embeddings = pd.concat([ims_embeddings, mass_spec_embeddings], axis=1)
# all_true_embeddings.head()
#%%
sorted_chem_names = sorted([chem for chem in name_smiles_embedding_df.index[1:]])
file_path = '../../data/encoder_embedding_predictions/ims_to_onehot_encoder_test_preds.csv'
test_embeddings = pd.read_csv(file_path)
indices = [idx for idx in test_embeddings.index]
chem_names = [sorted_chem_names[list(encoding).index(1)] for encoding in test_embeddings.iloc[:,-8:].values]
test_embeddings.insert(1, 'Label', chem_names)

y_test, x_test, test_chem_encodings_tensor, test_indices_tensor = f.create_dataset_tensors(
    test_embeddings, name_smiles_embedding_df, device, start_idx=2, stop_idx=-8
    )
#%%
encoder_path = '../trained_models/onehot_to_chemnet_encoder.pth'
best_model = torch.load(encoder_path, weights_only=False)
encoder_criterion = nn.MSELoss()
batch_size = 32

test_dataset = DataLoader(
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
    test_dataset, best_model, device, encoder_criterion)
input_indices = [idx for idx_list in input_indices for idx in idx_list]
predicted_embeddings = [emb for emb_list in predicted_embeddings for emb in emb_list]
output_name_encodings = [enc for enc_list in output_name_encodings for enc in enc_list]
test_preds_df = pd.DataFrame(predicted_embeddings)
test_preds_df.insert(0, 'index', input_indices)
name_encodings_df = pd.DataFrame(output_name_encodings)
name_encodings_df.columns = sorted_chem_names
test_preds_df = pd.concat([test_preds_df, name_encodings_df], axis=1)

file_path = '../../data/encoder_embedding_predictions/onehot_to_chemnet_encoder_test_preds.csv'
test_preds_df.to_csv(file_path, index=False)
#%%


file_path = '../../data/encoder_embedding_predictions/onehot_to_chemnet_encoder_test_preds.csv'
test_preds_df = pd.read_csv(file_path)
test_preds_df.head()
sorted_chem_names = ['DEB','DEM','DMMP','DPM','DtBP','JP8','MES','TEPO']
encodings_list = test_preds_df[sorted_chem_names].values.tolist()
embeddings_only = test_preds_df.iloc[:,1:-8]
embeddings_only.columns = ims_embeddings.T.columns
embeddings_only['Label'] = [sorted_chem_names[enc.index(1)] for enc in encodings_list]
#%%
importlib.reload(pf)
pf.plot_emb_pca(
    all_true_embeddings, embeddings_only, 'Test', 'IMS', 
    log_wandb=False, chemnet_embeddings_to_plot=ims_embeddings,
    show_wandb_run_name=False)