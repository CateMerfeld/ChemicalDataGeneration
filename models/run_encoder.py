# General script to run encoder model
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import os
import sys
import torch
import torch.nn as nn

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
# import plotting_functions as pf
import functions as f

#%%

# Things that need to be changed for each encoder/dataset/target embedding
notebook_name = '/home/cmdunham/ChemicalDataGeneration/models/run_encoder.py'
architecture = 'ims_to_ChemNet_encoder'
dataset_type = 'PHIL'
target_embedding = 'ChemNet'
model_type = 'Encoder'
loss = 'MSELoss'

early_stopping_threshold = 10
lr_scheduler_patience = 5
start_idx = 2
stop_idx = -9

# # generate_embeddings allows user to decide whether or not to generate embeddings based off of model performance. 
# # If set to None, user will be prompted to enter 'y' or 'n' after training.

# generate_embeddings = None
generate_embeddings = 'y'

# # best_hyperparameters is the output of training the encoder.
# # If running this file for prediction only (using a pre-trained model),
# # set best_hyperparameters['batch_size'] to the batch size used to train the model. 
# best_hyperparams = {'batch_size':16}

model_hyperparams = {
  'batch_size':[16],#, 32],
  'epochs': [100],
  'learning_rate':[.00001]#,, .000001],
  }

encoder_criterion = nn.MSELoss()

scaling_factor = 10


encoder_save_path = f'trained_models/{dataset_type}/{scaling_factor}_pct_scaling.pth'
name_smiles_embedding_df_file_path = '../../scratch/name_smiles_embedding_file.csv'
mass_spec_name_smiles_embedding_df_file_path = '../data/mass_spec_name_smiles_embedding_file.csv'
train_file_path = f'../../scratch/PHIL/train_phils_scaled_to_{scaling_factor}_pct.csv'
val_file_path = f'../../scratch/PHIL/val_phils_scaled_to_{scaling_factor}_pct.csv'
test_file_path = f'../../scratch/PHIL/test_phils_scaled_to_{scaling_factor}_pct.csv'
# train_file_path = '../../../scratch/train_data.feather'
# val_file_path = '../../../scratch/val_data.feather'
# test_file_path = '../../../scratch/test_data.feather'
train_preds_file_path = f'../../scratch/{dataset_type}/train_embedding_preds_scaled_to_{scaling_factor}_pct.feather'
val_preds_file_path = f'../../scratch/{dataset_type}/val_embedding_preds_scaled_to_{scaling_factor}_pct.feather'
test_preds_file_path = f'../../scratch/{dataset_type}/test_embedding_preds_scaled_to_{scaling_factor}_pct.feather'


sorted_chem_names = ['DEB','DEM','DMMP','DPM','DtBP','JP8','MES','TEPO']
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
#%%
# Loading Data:
name_smiles_embedding_df = f.format_embedding_df(name_smiles_embedding_df_file_path)

embeddings_df = pd.DataFrame([emb for emb in name_smiles_embedding_df['Embedding Floats']][1:]).T
cols = name_smiles_embedding_df.index[1:]
embeddings_df.columns = cols
#%%
train_data = pd.read_csv(train_file_path)
y_train, x_train, train_chem_encodings_tensor, train_indices_tensor = f.create_dataset_tensors(
    train_data, name_smiles_embedding_df, device, start_idx=start_idx, stop_idx=stop_idx)
sorted_chem_names = list(train_data.columns[-8:])
del train_data

# %%
val_data = pd.read_csv(val_file_path)
y_val, x_val, val_chem_encodings_tensor, val_indices_tensor = f.create_dataset_tensors(
    val_data, name_smiles_embedding_df, device, start_idx=start_idx, stop_idx=stop_idx)
del val_data
#%%
test_data = pd.read_csv(test_file_path)
y_test, x_test, test_chem_encodings_tensor, test_indices_tensor = f.create_dataset_tensors(
    test_data, name_smiles_embedding_df, device, start_idx=start_idx, stop_idx=stop_idx)
del test_data
# %%

train_data = TensorDataset(x_train, train_chem_encodings_tensor, y_train, train_indices_tensor)
val_data = TensorDataset(x_val, val_chem_encodings_tensor, y_val, val_indices_tensor)
test_data = TensorDataset(x_test, test_chem_encodings_tensor, y_test, test_indices_tensor)


best_hyperparams = f.train_model(
    model_type, train_data, val_data, test_data,
    device, config, wandb_kwargs,
    embeddings_df , name_smiles_embedding_df, model_hyperparams, 
    sorted_chem_names, encoder_save_path, save_emb_pca_to_wandb=True,
    input_type=dataset_type, embedding_type=target_embedding,
    early_stop_threshold=early_stopping_threshold, lr_scheduler=True, patience=lr_scheduler_patience,
    )

#%%
if generate_embeddings == None:
    generate_embeddings = f.get_input()

if generate_embeddings == 'y':
    batch_size = best_hyperparams['batch_size']

    mass_spec_name_smiles_embedding_df = f.format_embedding_df(mass_spec_name_smiles_embedding_df_file_path)
    mass_spec_embeddings_df = pd.DataFrame([emb for emb in mass_spec_name_smiles_embedding_df['Embedding Floats']]).T #mass_spec_embeddings[chems_above_5]
    cols = mass_spec_name_smiles_embedding_df.index
    mass_spec_embeddings_df.columns = cols

    ims_embeddings = pd.DataFrame([emb for emb in name_smiles_embedding_df['Embedding Floats']][1:]).T
    cols = name_smiles_embedding_df.index[1:]
    ims_embeddings.columns = cols
    all_true_embeddings = pd.concat([embeddings_df, mass_spec_embeddings_df], axis=1)

    best_model = torch.load(encoder_save_path, weights_only=False)

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
        train_data, best_model, device, encoder_criterion)

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
        val_data, best_model, device, encoder_criterion)

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
        test_data, best_model, device, encoder_criterion)

    test_preds_df = f.format_preds_df(
        input_indices, predicted_embeddings, 
        output_name_encodings, sorted_chem_names
        )

    test_preds_df.to_feather(test_preds_file_path)#, index=False)

else:
    print('Not generating embeddings.')
