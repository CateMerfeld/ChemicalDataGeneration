#%%
# Load Packages and Files:
import pandas as pd
#%%
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import os
import importlib
import functions as f
#%%
# Reload the functions module after updates
importlib.reload(f)

#%%

def run_generator(file_path_dict, chem, model_hyperparams, wandb_kwargs, sorted_chem_names, generator_path, notebook_name, num_plots):
    
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

    # remove from memory since information is now stored in train/val/test datasets
    # del train_embeddings_tensor, train_chem_encodings_tensor, train_carl_tensor, train_carl_indices_tensor
    # del val_embeddings_tensor, val_chem_encodings_tensor, val_carl_tensor, val_carl_indices_tensor
    # del test_embeddings_tensor, test_chem_encodings_tensor, test_carl_tensor, test_carl_indices_tensor

    config = {
        'wandb_entity': 'catemerfeld',
        'wandb_project': 'ims_encoder_decoder',
        'gpu':True,
        'threads':1,
    }

    os.environ['WANDB_NOTEBOOK_NAME'] = notebook_name

    f.train_generator(
        train_data, val_data, test_data, device, config, 
        wandb_kwargs, model_hyperparams, sorted_chem_names, 
        generator_path, early_stop_threshold=wandb_kwargs['early stopping threshold'], 
        lr_scheduler=True, num_plots=num_plots, plot_overlap_pca=True
        )

#%%

sorted_chem_names = ['DEB','DEM','DMMP','DPM','DtBP','JP8','MES','TEPO']
file_path_dict = {'train_carls_file_path':'../data/carls/train_carls_one_per_spec.feather', 
                  'train_embeddings_file_path':'../data/encoder_embedding_predictions/reparameterization_train_preds.csv',
                #   '../data/encoder_embedding_predictions/train_embeddings.csv',

                  'val_carls_file_path': '../data/carls/val_carls_one_per_spec.feather', 
                  'val_embeddings_file_path': '../data/encoder_embedding_predictions/reparameterization_val_preds.csv',
                #   '../data/encoder_embedding_predictions/val_embeddings.csv',

                  'test_carls_file_path':'../data/carls/test_carls_one_per_spec.feather', 
                  'test_embeddings_file_path':  '../data/encoder_embedding_predictions/reparameterization_test_preds.csv',
                #   '../data/encoder_embedding_predictions/test_embeddings.csv'
                }
# #%%
# wandb_kwargs = {
#     'architecture': 'individual_carl_generator',
#     'optimizer':'AdamW',
#     'loss':'MSELoss',
#     'dataset': 'carls',
#     'target_embedding': 'ChemNet',
#     'early stopping threshold':20
# }
# model_hyperparams = {
#     'batch_size':[4,16],
#     'epochs': [500],
#     'learning_rate':[.001,.0001],
#     }
# notebook_name = '/home/cmdunham/ChemicalDataGeneration/models/individual_generator.py'
# num_plots = 5
# chem_to_run='MES'

# for chem in sorted_chem_names:
#     generator_path= f'../models/trained_models/{chem}_carl_to_chemnet_generator_reparameterization.pth'
#     wandb_kwargs['analyte']=chem
#     if chem == chem_to_run:
#         run_generator(file_path_dict, chem, model_hyperparams, wandb_kwargs, sorted_chem_names, generator_path, notebook_name, num_plots)
# #%%

# # Get predictions and save synthetic spectra
# chem_to_run='MES'
# batch_size = 4
# criterion = nn.MSELoss()
# num_plots = 5

# device = f.set_up_gpu()
# generator_path = f'../models/trained_models/{chem_to_run}_carl_to_chemnet_generator_reparameterization.pth'
# best_model = torch.load(generator_path, weights_only=False)
# # train_embeddings_tensor, train_carl_tensor, train_chem_encodings_tensor, train_carl_indices_tensor = f.create_individual_chemical_dataset_tensors(
# # file_path_dict['train_carls_file_path'], file_path_dict['train_embeddings_file_path'], device, chem, multiple_carls_per_spec=False
# # )
# # val_embeddings_tensor, val_carl_tensor, val_chem_encodings_tensor, val_carl_indices_tensor = f.create_individual_chemical_dataset_tensors(
# # file_path_dict['val_carls_file_path'], file_path_dict['val_embeddings_file_path'], device, chem, multiple_carls_per_spec=False
# # )
# test_embeddings_tensor, test_carl_tensor, test_chem_encodings_tensor, test_carl_indices_tensor = f.create_individual_chemical_dataset_tensors(
# file_path_dict['test_carls_file_path'], file_path_dict['test_embeddings_file_path'], device, chem_to_run, multiple_carls_per_spec=False
# )
# #%%
# test_carls = pd.read_feather(file_path_dict['test_carls_file_path'])
# # print(test_carls.head())
# #%%
# # train_data = TensorDataset(train_embeddings_tensor, train_chem_encodings_tensor, train_carl_tensor, train_carl_indices_tensor)
# # val_data = TensorDataset(val_embeddings_tensor, val_chem_encodings_tensor, val_carl_tensor, val_carl_indices_tensor)
# test_data = TensorDataset(test_embeddings_tensor, test_chem_encodings_tensor, test_carl_tensor, test_carl_indices_tensor)
# #%%
# # file_path = '../../scratch/test_data.feather'
# # test_spectra = pd.read_feather(file_path)
# #%%
# file_path = '../../scratch/test_avg_bkg.csv'
# test_avg_bkg = pd.read_csv(file_path)
# test_avg_bkg.drop(columns=['Unnamed: 0'], inplace=True)

# #%%
# test_dataset = DataLoader(test_data, batch_size)
# test_predicted_carls, test_output_name_encodings, _, test_spectra_indices = f.predict_embeddings(test_dataset, best_model, device, criterion)
# #%%

# # combine all the predicted carls and corresponding background spectra to create synthetic spectra
# preds_list = [pred for pred_list in test_predicted_carls for pred in pred_list]
# indices_list = [ind for ind_list in test_spectra_indices for ind in ind_list]

# synthetic_spectra = []

# for pred in preds_list:
#     synthetic_spec = pred + test_avg_bkg
#     synthetic_spectra.append(synthetic_spec.values.flatten())

# # Create list of chemical names for generated spectra
# test_chem_encodings_list = [enc for enc_list in test_output_name_encodings for enc in enc_list]
# test_labels = [sorted_chem_names[list(enc).index(1)] for enc in test_chem_encodings_list]

# synthetic_spectra_df = pd.DataFrame(synthetic_spectra)
# synthetic_spectra_df['Label'] = test_labels
# synthetic_spectra_df.insert(0, 'index', indices_list)
# synthetic_spectra_df.columns = test_carls.columns[1:-8]
# #%%
# file_path = f'../data/ims_data/synthetic_test_{chem_to_run}_spectra_reparameterization.csv'
# synthetic_spectra_df.to_csv(file_path)
#%%
