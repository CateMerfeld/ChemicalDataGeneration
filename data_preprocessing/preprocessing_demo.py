#%%
# import requests
from fcd_torch import FCD
import torch
import pandas as pd
import numpy as np
#%%
import os
import sys
#%%
import GPUtil
#%%
def reformat_spectra_df(df):
    """
    Transforms the 'Spectrum' column of a DataFrame into a new DataFrame where
    the 'SMILES' column is used as column headers and the formatted 'Spectrum'
    column becomes the data.

    Parameters:
    df (pd.DataFrame): Input DataFrame with 'SMILES' and 'Spectrum' columns.

    Returns:
    pd.DataFrame: Transformed DataFrame.
    """
    transformed_data = {}

    for _, row in df.iterrows():
        smiles = row['SMILES']
        spectrum = row['Spectrum'].split(' ')

        if spectrum[0][0] == 'k':
            spectrum = spectrum[1:]

        indices = [round(float(pair.split(':')[0])) for pair in spectrum]
        values = [float(pair.split(':')[1]) for pair in spectrum]

        max_index = max(indices)
        spectrum_df = pd.DataFrame(np.zeros((max_index + 1, 1)), columns=[smiles])
        for idx, val in zip(indices, values):
            spectrum_df.at[idx, smiles] = val

        transformed_data[smiles] = spectrum_df[smiles]

    formatted_df = pd.concat(transformed_data.values(), axis=1)
    formatted_df = formatted_df.fillna(0)
    return formatted_df

# Example usage
df = pd.DataFrame({
    'SMILES': ['CCO', 'CCN'],
    'Spectrum': ['k3.4:125 4.6:857 13.9:20.3', '2.1:50 5.5:100 10.2:75']
})

result_df = reformat_spectra_df(df)
#%%

#%%
def get_chemnet_emb_from_smiles(smiles_list):
    """
    Get ChemNet embeddings from a list of SMILES strings.

    Parameters:
    smiles_list (list): List of SMILES strings.

    Returns:
    dict: A dictionary mapping each SMILES string to its corresponding ChemNet embedding.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fcd = FCD(device, n_jobs=1)
    
    smiles_emb_dict = {}

    for smiles in smiles_list:
        try:
            emb = fcd.get_predictions([smiles])[0]
            smiles_emb_dict[smiles] = list(emb)
        except KeyError as e:
            if e == 'PropertyTable':
                smiles_emb_dict[smiles] = 'unknown'

    return smiles_emb_dict

chemical_list = ['MES', 'DtBP']
smiles_list = ['C1COCCN1CCS(=O)(=O)O', 'CC(C)(C)OOC(C)(C)C'] # MES and DtBP SMILES
smiles_emb_dict = get_chemnet_emb_from_smiles(smiles_list)
for chem, smiles in zip(chemical_list, smiles_list):
    print('The ChemNet embedding for {} is {} dimensions and begins with:'.format(
        chem, 
        len(smiles_emb_dict[smiles]),
    ))
    print(smiles_emb_dict[smiles][:5])
    print('')

#%%
def set_up_gpu():
    if torch.cuda.is_available():
        # Get the list of GPUs
        gpus = GPUtil.getGPUs()

        # Find the GPU with the most free memory
        best_gpu = max(gpus, key=lambda gpu: gpu.memoryFree)

        # Print details about the selected GPU
        print(f"Selected GPU ID: {best_gpu.id}")
        print(f"  Name: {best_gpu.name}")
        print(f"  Memory Free: {best_gpu.memoryFree} MB")
        print(f"  Memory Used: {best_gpu.memoryUsed} MB")
        print(f"  GPU Load: {best_gpu.load * 100:.2f}%")

        # Set the device for later use
        device = torch.device(f'cuda:{best_gpu.id}')
        print('Current device ID: ', device)

        # Set the current device in PyTorch
        torch.cuda.set_device(best_gpu.id)
    else:
        device = torch.device('cpu')
        print('Using CPU')
        

    # Confirm the currently selected device in PyTorch
    print("PyTorch current device ID:", torch.cuda.current_device())
    print("PyTorch current device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

    return device

# device = set_up_gpu()
smiles_list = ['O=C(O)C(Cl)Cl', 'CC(=O)C(Cl)Cl']
emb_dict = get_chemnet_emb_from_smiles(smiles_list)
print(emb_dict.keys())
print(len(emb_dict['O=C(O)C(Cl)Cl']))