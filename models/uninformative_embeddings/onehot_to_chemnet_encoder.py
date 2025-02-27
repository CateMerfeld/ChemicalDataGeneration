#%%
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset
import os
#%%
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import plotting_functions as pf
import functions as f
#%%
file_path = '../../data/name_smiles_embedding_file.csv'
name_smiles_embedding_df = f.format_embedding_df(file_path)
name_smiles_embedding_df.head()
#%%
sorted_chem_names = sorted([chem for chem in name_smiles_embedding_df.index[1:]])
true_embeddings = pd.get_dummies(sorted_chem_names, dtype=np.float32)
# indices are unnecessary here but are needed for the create_dataset_tensors function
indices = [idx for idx in true_embeddings.index]
true_embeddings.insert(0, 'index', indices)
true_embeddings.insert(1, 'Label', sorted_chem_names)
# true_embeddings.head()
#%%
file_path = '../../data/encoder_embedding_predictions/ims_to_onehot_encoder_val_preds.csv'
val_embeddings = pd.read_csv(file_path)
indices = [idx for idx in val_embeddings.index]
chem_names = [sorted_chem_names[list(encoding).index(1)] for encoding in val_embeddings.iloc[:,-8:].values]
val_embeddings.insert(1, 'Label', chem_names)
#%%
file_path = '../../data/encoder_embedding_predictions/ims_to_onehot_encoder_test_preds.csv'
test_embeddings = pd.read_csv(file_path)
indices = [idx for idx in test_embeddings.index]
chem_names = [sorted_chem_names[list(encoding).index(1)] for encoding in test_embeddings.iloc[:,-8:].values]
test_embeddings.insert(1, 'Label', chem_names)
#%%
device = f.set_up_gpu()
y_train, x_train, train_chem_encodings_tensor, train_carl_indices_tensor = f.create_dataset_tensors(
    true_embeddings, name_smiles_embedding_df, device, start_idx=2
    )
y_val, x_val, val_chem_encodings_tensor, val_carl_indices_tensor = f.create_dataset_tensors(
    val_embeddings, name_smiles_embedding_df, device, start_idx=2, stop_idx=-8
    )
y_test, x_test, test_chem_encodings_tensor, test_carl_indices_tensor = f.create_dataset_tensors(
    test_embeddings, name_smiles_embedding_df, device, start_idx=2, stop_idx=-8
    )


#%%

# Things that need to be changed for each encoder/dataset/target embedding
notebook_name = '/home/cmdunham/ChemicalDataGeneration/models/uninformative_embeddings/onehot_to_chemnet_encoder.py'
architecture = 'onehot_to_chemnet_encoder'
dataset_type = 'OneHot'
target_embedding = 'ChemNet'
encoder_path = '../trained_models/onehot_to_chemnet_encoder.pth'
model_type = 'OneHottoChemNetEncoder'
loss = 'MSELoss'

config = {
    'wandb_entity': 'catemerfeld',
    'wandb_project': 'ims_encoder_decoder',
    'gpu':True,
    'threads':1,
}

os.environ['WANDB_NOTEBOOK_NAME'] = notebook_name

early_stopping_threshold = 15
wandb_kwargs = {
    'architecture': architecture,
    'optimizer':'AdamW',
    'loss': loss,
    'dataset': dataset_type,
    'target_embedding': target_embedding,
    'early stopping threshold':early_stopping_threshold
}

model_hyperparams = {
  'batch_size':[32],
  'epochs': [1000],
  'learning_rate':[.0001],
  }
# print(x_train)
train_data = TensorDataset(x_train, train_chem_encodings_tensor, y_train, train_carl_indices_tensor)
val_data = TensorDataset(x_val, val_chem_encodings_tensor, y_val, val_carl_indices_tensor)
test_data = TensorDataset(x_test, test_chem_encodings_tensor, y_test, test_carl_indices_tensor)

ims_embeddings = pd.DataFrame([emb for emb in name_smiles_embedding_df['Embedding Floats']][1:]).T
cols = name_smiles_embedding_df.index[1:]
ims_embeddings.columns = cols

best_hyperparams = f.train_model(
    model_type, train_data, train_data, train_data, 
    device, config, wandb_kwargs, ims_embeddings, 
    name_smiles_embedding_df, model_hyperparams, sorted_chem_names, 
    encoder_path, save_emb_pca_to_wandb=True, 
    early_stop_threshold=early_stopping_threshold,
    input_type=dataset_type, embedding_type=target_embedding,
    show_wandb_run_name=True, lr_scheduler=True
    )