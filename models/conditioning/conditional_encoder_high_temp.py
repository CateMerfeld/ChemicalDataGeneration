#%%
import pandas as pd
#%%
from torch.utils.data import TensorDataset
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
# import plotting_functions as pf
import functions as f
#%%
import importlib
# # Reload the functions module after updates
importlib.reload(f)
#%%
file_path = '../../data/name_smiles_embedding_file.csv'
name_smiles_embedding_df = f.format_embedding_df(file_path)
# print(name_smiles_embedding_df.head())

device = f.set_up_gpu()

start_idx = 2
stop_idx = -9
# %%
# Loading Data:
file_path = '../../data/train_test_val_splits/train_carls_high_TemperatureKelvin.csv'
train_carls = pd.read_csv(file_path)
train_carls['Label'].value_counts()
y_train, x_train, train_chem_encodings_tensor, train_carl_indices_tensor = f.create_dataset_tensors(
    train_carls, name_smiles_embedding_df, device, start_idx=start_idx, stop_idx=stop_idx)
sorted_chem_names = list(train_carls.columns[-8:])
del train_carls

#%%
file_path = '../../data/train_test_val_splits/val_carls_high_TemperatureKelvin.csv'
val_carls = pd.read_csv(file_path)

y_val, x_val, val_chem_encodings_tensor, val_carl_indices_tensor = f.create_dataset_tensors(
    val_carls, name_smiles_embedding_df, device, start_idx=start_idx, stop_idx=stop_idx)
del val_carls
#%%
file_path = '../../data/train_test_val_splits/test_carls_high_TemperatureKelvin.csv'
test_carls = pd.read_csv(file_path)
y_test, x_test, test_chem_encodings_tensor, test_carl_indices_tensor = f.create_dataset_tensors(
    test_carls, name_smiles_embedding_df, device, start_idx=start_idx, stop_idx=stop_idx)
del test_carls

# %%

file_path = '../../data/mass_spec_name_smiles_embedding_file.csv'
mass_spec_name_smiles_embedding_df = f.format_embedding_df(file_path)
# %%
# Combine embeddings for IMS simulants and mass spec chems to use for plotting pca
ims_embeddings = pd.DataFrame([emb for emb in name_smiles_embedding_df['Embedding Floats']][1:]).T
mass_spec_embeddings = pd.DataFrame([emb for emb in mass_spec_name_smiles_embedding_df['Embedding Floats']]).T
cols = name_smiles_embedding_df.index[1:]
ims_embeddings.columns = cols
cols = mass_spec_name_smiles_embedding_df.index
mass_spec_embeddings.columns = cols
all_true_embeddings = pd.concat([ims_embeddings, mass_spec_embeddings], axis=1)
# %%
# Training Encoder on Carls:

# Things that need to be changed for each encoder/dataset/target embedding
notebook_name = '/home/cmdunham/ChemicalDataGeneration/models/conditional_encoder_high_temp.py'
architecture = 'carl_encoder'
dataset_type = 'high_temp_carls'
target_embedding = 'ChemNet'
encoder_type=''
data_condition = 'high'
encoder_path = f'../trained_models/{data_condition}_temp_carl_to_chemnet_encoder{encoder_type}.pth'
input_type = 'Carl'
model_type = 'Encoder'

config = {
    'wandb_entity': 'catemerfeld',
    'wandb_project': 'ims_encoder_decoder',
    'gpu':True,
    'threads':1,
}

os.environ['WANDB_NOTEBOOK_NAME'] = notebook_name

# Reload the functions module after updates
importlib.reload(f)
early_stopping_threshold = 20
wandb_kwargs = {
    'architecture': architecture,
    'optimizer':'AdamW',
    'loss':'MSELoss',
    'dataset': dataset_type,
    'target_embedding': target_embedding,
    'early stopping threshold':early_stopping_threshold
}

model_hyperparams = {
  'batch_size':[32],
  'epochs': [500],
  'learning_rate':[.0001],
  }

train_data = TensorDataset(x_train, train_chem_encodings_tensor, y_train, train_carl_indices_tensor)
val_data = TensorDataset(x_val, val_chem_encodings_tensor, y_val, val_carl_indices_tensor)
test_data = TensorDataset(x_test, test_chem_encodings_tensor, y_test, test_carl_indices_tensor)

best_hyperparams = f.train_model(
    model_type, train_data, val_data, test_data, 
    device, config, wandb_kwargs, 
    all_true_embeddings, name_smiles_embedding_df, model_hyperparams, 
    sorted_chem_names, encoder_path, save_emb_pca_to_wandb=True, early_stop_threshold=early_stopping_threshold,
    input_type=input_type, show_wandb_run_name=True, lr_scheduler=True
    )

# %%
# chem_list = [
#     '(+)-Borneol', 'Acetophenone', 'Adenine', 'Adenosine', 'Allantoin',
#     'Amantadine', 'Anthracene', 'DIETHYL MALEATE', 'Diethyl fumarate',
#     'Fenchol', 'Fumaric acid', 'Glutaric acid', 'Indole-3-acetic acid',
#     'Isonicotinic acid', 'L-Asparagine', 'L-Aspartic acid',
#     'L-Glutamic acid', 'L-Glutamine', 'L-Iditol', 'L-Isoleucine',
#     'L-Lysine', 'L-Norleucine', 'L-Serine', 'L-Valine',
#     'METHYL PROPIONATE', 'Malonic acid', 'Maltotriose', 'Melibiose',
#     'Methyl hexanoate', 'N-Acetyl-D-glucosamine',
#     'N-Formyl-L-Methionine', 'Pyrene', 'Sinapic acid', 'Spermidine',
#     'Spermine', 'Succinic acid', 'Testosterone', 'beta-Alanine',
#     'cis-Citral', 'trans-Cinnamyl alcohol'
#     ]
# #%%
# importlib.reload(f)
# file_path = '../../data/mass_spec_name_smiles_embedding_file.csv'
# mass_spec_name_smiles_embedding_df = f.format_embedding_df(file_path, chem_list=chem_list)