#%%
# Load Packages and Files:
import pandas as pd
#%%
# import matplotlib.pyplot as plt
# import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset 
import torch.nn as nn
# import wandb
import os
# from collections import Counter
import importlib
import functions as f
# import GPUtil

#%%
# Reload the functions module after updates
importlib.reload(f)

#%%


def run_generator(file_path_dict, chem, model_hyperparams, sorted_chem_names, early_stopping_threshold=20):
    device = f.set_up_gpu()
    train_embeddings_tensor, train_carl_tensor, train_chem_encodings_tensor, train_carl_indices_tensor = f.create_individual_chemical_dataset_tensors(
    file_path_dict['train_carls_file_path'], file_path_dict['train_embeddings_file_path'], device, chem, multiple_carls_per_spec=False
    )
    val_embeddings_tensor, val_carl_tensor, val_chem_encodings_tensor, val_carl_indices_tensor = f.create_individual_chemical_dataset_tensors(
    file_path_dict['val_carls_file_path'], file_path_dict['val_embeddings_file_path'], device, chem, multiple_carls_per_spec=False
    )
    test_embeddings_tensor, test_carl_tensor, test_chem_encodings_tensor, test_carl_indices_tensor = f.create_individual_chemical_dataset_tensors(
    file_path_dict['test_carls_file_path'], file_path_dict['test_embeddings_file_path'], device, chem, multiple_carls_per_spec=False
    )

    train_data = TensorDataset(train_embeddings_tensor, train_chem_encodings_tensor, train_carl_tensor, train_carl_indices_tensor)
    val_data = TensorDataset(val_embeddings_tensor, val_chem_encodings_tensor, val_carl_tensor, val_carl_indices_tensor)
    test_data = TensorDataset(test_embeddings_tensor, test_chem_encodings_tensor, test_carl_tensor, test_carl_indices_tensor)

    notebook_name = '/home/cmdunham/ChemicalDataGeneration/models/individual_ims_generator.ipynb'
    architecture = 'individual_carl_generator'
    dataset_type = 'carls'
    target_embedding = 'ChemNet'
    generator_path = f'../models/trained_models/{chem}_carl_to_chemnet_generator.pth'

    config = {
        'wandb_entity': 'catemerfeld',
        'wandb_project': 'ims_encoder_decoder',
        'gpu':True,
        'threads':1,
    }

    os.environ['WANDB_NOTEBOOK_NAME'] = notebook_name

    wandb_kwargs = {
        'architecture': architecture,
        'analyte':chem,
        'optimizer':'AdamW',
        'loss':'MSELoss',
        'dataset': dataset_type,
        'target_embedding': target_embedding,
        'early stopping threshold':early_stopping_threshold
    }

    f.train_generator(
        train_data, val_data, test_data, device, config, 
        wandb_kwargs, model_hyperparams, sorted_chem_names, 
        generator_path, early_stop_threshold=early_stopping_threshold, 
        lr_scheduler=True, num_plots=5, plot_overlap_pca=True
        )

#%%
# model_hyperparams = {
#     'batch_size':[16,32],
#     'epochs': [500],
#     'learning_rate':[.001, .0001],
#     }
model_hyperparams = {
    'batch_size':[4],
    'epochs': [500],
    'learning_rate':[.001],
    }

#%%
sorted_chem_names = ['DEB','DEM','DMMP','DPM','DtBP','JP8','MES','TEPO']
# sorted_chem_names = ['JP8']#,'TEPO']
file_path_dict = {'train_carls_file_path':'../data/carls/train_carls_one_per_spec.feather', 
                  'train_embeddings_file_path':'../data/encoder_embedding_predictions/train_embeddings.csv',
                  'val_carls_file_path':'../data/carls/val_carls_one_per_spec.feather', 
                  'val_embeddings_file_path':'../data/encoder_embedding_predictions/val_embeddings.csv',
                  'test_carls_file_path':'../data/carls/test_carls_one_per_spec.feather', 
                  'test_embeddings_file_path':'../data/encoder_embedding_predictions/test_embeddings.csv'}
# #%%
# for chem in sorted_chem_names:
#     if chem == 'JP8': # or chem == 'MES':
#         run_generator(file_path_dict, chem, model_hyperparams, sorted_chem_names)
#%%
device = f.set_up_gpu()
generator_path = '../models/trained_models/JP8_carl_to_chemnet_generator.pth'
best_model = torch.load(generator_path, weights_only=False)
# f.Generator().to(device)
# best_model.load(torch.load(generator_path))
# print(best_model)
chem='JP8'
# train_embeddings_tensor, train_carl_tensor, train_chem_encodings_tensor, train_carl_indices_tensor = f.create_individual_chemical_dataset_tensors(
# file_path_dict['train_carls_file_path'], file_path_dict['train_embeddings_file_path'], device, chem, multiple_carls_per_spec=False
# )
# val_embeddings_tensor, val_carl_tensor, val_chem_encodings_tensor, val_carl_indices_tensor = f.create_individual_chemical_dataset_tensors(
# file_path_dict['val_carls_file_path'], file_path_dict['val_embeddings_file_path'], device, chem, multiple_carls_per_spec=False
# )
test_embeddings_tensor, test_carl_tensor, test_chem_encodings_tensor, test_carl_indices_tensor = f.create_individual_chemical_dataset_tensors(
file_path_dict['test_carls_file_path'], file_path_dict['test_embeddings_file_path'], device, chem, multiple_carls_per_spec=False
)
#%%
test_carls = pd.read_feather(file_path_dict['test_carls_file_path'])
#%%
# train_data = TensorDataset(train_embeddings_tensor, train_chem_encodings_tensor, train_carl_tensor, train_carl_indices_tensor)
# val_data = TensorDataset(val_embeddings_tensor, val_chem_encodings_tensor, val_carl_tensor, val_carl_indices_tensor)
test_data = TensorDataset(test_embeddings_tensor, test_chem_encodings_tensor, test_carl_tensor, test_carl_indices_tensor)
#%%
# file_path = '../../scratch/test_data.feather'
# test_spectra = pd.read_feather(file_path)
#%%
file_path = '../../scratch/test_avg_bkg.csv'
test_avg_bkg = pd.read_csv(file_path)
test_avg_bkg.drop(columns=['Unnamed: 0'], inplace=True)

#%%
batch_size = 16
criterion = nn.MSELoss()
num_plots = 5

test_dataset = DataLoader(test_data, batch_size)
test_predicted_carls, test_output_name_encodings, _, _ = f.predict_embeddings(test_dataset, best_model, device, criterion)
#%%

# combine all the predicted carls and corresponding background spectra to create synthetic spectra
preds_list = [pred for pred_list in test_predicted_carls for pred in pred_list]

synthetic_spectra = []

for pred in preds_list:
    synthetic_spec = pred + test_avg_bkg
    synthetic_spectra.append(synthetic_spec.values.flatten())

# Create list of chemical names for generated spectra
test_chem_encodings_list = [enc for enc_list in test_output_name_encodings for enc in enc_list]
test_labels = [sorted_chem_names[list(enc).index(1)] for enc in test_chem_encodings_list]

synthetic_spectra_df = pd.DataFrame(synthetic_spectra)
synthetic_spectra_df['Label'] = test_labels
synthetic_spectra_df.columns = test_carls.columns[2:-8]
#%%
file_path = '../data/ims_data/synthetic_test_spectra.csv'
synthetic_spectra_df.to_csv(file_path)