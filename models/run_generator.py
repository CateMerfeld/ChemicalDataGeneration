# General script to run generator model

# Load Packages and Files:
import pandas as pd
#%%
# import numpy as np
# import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
# import os
import importlib
import functions as f
import time
#%%
# Reload the functions module after updates
importlib.reload(f)

#%%
# best_hyperparameters = {'batch_size':16}
notebook_name = '/home/cmdunham/ChemicalDataGeneration/models/run_generator.py'
architecture = 'universal_spectrum_generator'
loss = 'MSELoss'
criterion = nn.MSELoss()
input_type = 'carl_embeddings'
target_type = 'spectrum'
early_stopping_threshold = 15
num_plots = 5
# model_type = 'individual_generators'

start_idx = 2
stop_idx = -9
device = f.set_up_gpu()

model_hyperparams = {
    'batch_size':[16],
    'epochs': [50],
    'learning_rate':[.01],
    }

wandb_kwargs = {
    'architecture': architecture,
    'optimizer':'AdamW',
    'loss':loss,
    'input': input_type,
    'target': target_type,
    'early stopping threshold':early_stopping_threshold
}

# generator_save_path_pt_1 = f'trained_models/{target_type}/'
# generator_save_path_pt_2 = f'{architecture}.pth'
generator_save_path = f'trained_models/{target_type}/{architecture}.pth'
synthetic_data_save_path_pt_1 = f'../../scratch/synthetic_data/{target_type}/{architecture}/'
synthetic_data_save_path_pt_2 = 'synthetic_test_spectra.csv'
# train_file_path = '../data/carls/train_carls_one_per_spec.feather'
train_file_path = '../../scratch/train_data.feather'
train_embeddings_file_path = '../data/encoder_embedding_predictions/train_embeddings.csv'
# val_file_path = '../data/carls/val_carls_one_per_spec.feather'
val_file_path = '../../scratch/val_data.feather'
val_embeddings_file_path = '../data/encoder_embedding_predictions/val_embeddings.csv'
# test_file_path = '../data/carls/test_carls_one_per_spec.feather'
test_file_path = '../../scratch/test_data.feather'
test_embeddings_file_path = '../data/encoder_embedding_predictions/test_embeddings.csv'
test_avg_bkg_file_path = '../../scratch/test_avg_bkg.csv'

sorted_chem_names = ['DEB','DEM','DMMP','DPM','DtBP','JP8','MES','TEPO']

config = {
    'wandb_entity': 'catemerfeld',
    'wandb_project': 'ims_encoder_decoder',
    'gpu':True,
    'threads':1,
}

# chem_groups = [['DMMP', 'TEPO'], ['DEM', 'DPM', 'DEB'], ['DtBP', 'MES']]

# # If doing group generators this should be uncommented and next code blocks indented
# for group in chem_groups:
#     group_file_path = '_'.join(group)
#     generator_save_path = '_'.join([generator_save_path_pt_1,group_file_path,generator_save_path_pt_2])

    # wandb_kwargs['group']=group
# Loading data needs to be redone for each group because keeping the entire dataset in memory + the group subset is too much
train_data = pd.read_feather(train_file_path)
train_embeddings_df = pd.read_csv(train_embeddings_file_path)

# train_data = train_data[train_data[group].any(axis=1)] 
# train_embeddings_df = train_embeddings_df[train_embeddings_df[group].any(axis=1)] 

x_train, y_train, train_chem_encodings_tensor, train_indices_tensor = f.create_dataset_tensors_for_generator(
    train_data, train_embeddings_df, device, start_idx=start_idx, stop_idx=stop_idx)

del train_data, train_embeddings_df

#%%
val_data = pd.read_feather(val_file_path)
val_embeddings_df = pd.read_csv(val_embeddings_file_path)

# val_data = val_data[val_data[group].any(axis=1)] 
# val_embeddings_df = val_embeddings_df[val_embeddings_df[group].any(axis=1)]

x_val, y_val, val_chem_encodings_tensor, val_indices_tensor = f.create_dataset_tensors_for_generator(
    val_data, val_embeddings_df, device, start_idx=start_idx, stop_idx=stop_idx)
del val_data, val_embeddings_df
# %%
test_data = pd.read_feather(test_file_path)
test_embeddings_df = pd.read_csv(test_embeddings_file_path)

# test_data = test_data[test_data[group].any(axis=1)]
# test_embeddings_df = test_embeddings_df[test_embeddings_df[group].any(axis=1)] 

x_test, y_test, test_chem_encodings_tensor, test_indices_tensor = f.create_dataset_tensors_for_generator(
    test_data, test_embeddings_df, device, start_idx=start_idx, stop_idx=stop_idx)

del test_data, test_embeddings_df

#%%
train_data = TensorDataset(x_train, train_chem_encodings_tensor, y_train, train_indices_tensor)
val_data = TensorDataset(x_val, val_chem_encodings_tensor, y_val, val_indices_tensor)
test_data = TensorDataset(x_test, test_chem_encodings_tensor, y_test, test_indices_tensor)


best_hyperparameters = f.train_generator(
    train_data, val_data, test_data, device, config,
    wandb_kwargs, model_hyperparams, sorted_chem_names,
    generator_save_path, save_plots_to_wandb=False,
    early_stop_threshold=wandb_kwargs['early stopping threshold'], 
    num_plots=num_plots, # pretrained_model_path=generator_load_path,
    carl_or_spec=target_type
)
#%%

generate_synthetic_data = 'n'
print("Generate synthetic data using best trained model? (y/n)")
start_time = time.time()

while time.time() - start_time < 120:
    generate_synthetic_data = input()
    if generate_synthetic_data:
        break

if generate_synthetic_data == 'y':
    print('Generating synthetic data...')
    batch_size = best_hyperparameters['batch_size']

    # # If doing group generators this should be uncommented and next code blocks indented
    # for group in chem_groups:
        # group_file_path = '_'.join(group)
        # generator_path = '_'.join([generator_save_path_pt_1,group_file_path,generator_save_path_pt_2])
        # preds_save_path = '_'.join([synthetic_data_save_path_pt_1,group_file_path,synthetic_data_save_path_pt_2])
    generator_path = generator_save_path
    preds_save_path = '_'.join([synthetic_data_save_path_pt_1,synthetic_data_save_path_pt_2])

    test_data = pd.read_feather(test_file_path)
    test_embeddings_df = pd.read_csv(test_embeddings_file_path)

    # test_data = test_data[test_data[group].any(axis=1)]
    test_cols = test_data.columns[start_idx:stop_idx]
    # test_embeddings_df = test_embeddings_df[test_embeddings_df[group].any(axis=1)] 

    x_test, y_test, test_chem_encodings_tensor, test_indices_tensor = f.create_dataset_tensors_for_generator(
        test_data, test_embeddings_df, device, start_idx=start_idx, stop_idx=stop_idx)

    del test_data, test_embeddings_df

    test_data = TensorDataset(x_test, test_chem_encodings_tensor, y_test, test_indices_tensor)
    test_dataset = DataLoader(test_data, batch_size)

    model = f.load_model(generator_path, device=device)#, weights_only=False)
    test_preds, test_chem_name_encodings, _, test_indices = f.predict_embeddings(test_dataset, model, device, criterion)


    ##############
    if target_type == 'CARL':
        test_avg_bkg = pd.read_csv(test_avg_bkg_file_path)
        test_avg_bkg.drop(columns=['Unnamed: 0'], inplace=True)

        preds_list = [pred for pred_list in test_preds for pred in pred_list]
        indices_list = [ind for ind_list in test_indices for ind in ind_list]

        synthetic_spectra = []

        for pred in preds_list:
            synthetic_spec = pred + test_avg_bkg
            synthetic_spectra.append(synthetic_spec.values.flatten())

    # Create list of chemical names for generated spectra
    test_chem_name_encodings_list = [enc for enc_list in test_chem_name_encodings for enc in enc_list]
    test_labels = [sorted_chem_names[list(enc).index(1)] for enc in test_chem_name_encodings_list]

    synthetic_spectra = [pred for pred_list in test_preds for pred in pred_list]
    synthetic_spectra_df = pd.DataFrame(synthetic_spectra, columns=test_cols)
    synthetic_spectra_df['Label'] = test_labels
    synthetic_spectra_df.to_csv(preds_save_path, index=False)
else:
    print('Skipping synthetic data generation...')
