#%%
import functions as f
import plotting_functions as pf
import pandas as pd
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data_preprocessing'))
sys.path.append(parent_dir)
import preprocessing_functions as ppf
# from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
#%%
import importlib
importlib.reload(pf)
#%%

metadata = pd.read_feather('/home/cmdunham/scratch/BKG_SIM_ims_acbc_train_v1.1.09_meta.feather')

name_smiles_embedding_df_file_path = '/home/cmdunham/scratch/name_smiles_embedding_file.csv'
name_smiles_embedding_df = f.format_embedding_df(name_smiles_embedding_df_file_path)

ims_embeddings = pd.DataFrame([emb for emb in name_smiles_embedding_df['Embedding Floats']][1:]).T
cols = name_smiles_embedding_df.index[1:]
ims_embeddings.columns = cols
#%%
results_type = 'Test'
model_type = 'carl_to_chemnet_conditional'
# model_type = 'carl_to_chemnet'
# conditional variable used to create plot save path. Value should either be 'conditional_' or ''.
conditional = 'conditional_'
# whether to plot the predicted condition or not. If not, value should be an empty string.
plot_pred_condition = 'with_predicted_'
# in conditional encodings, insert the condition before the 513th column (named '512'), as we want to exclude the two conditional cols
# in non-conditional encodings, insert the condition before the 1st one-hot encoded column (named 'DEB)
insert_before = '512' if conditional == 'conditional_' else 'DEB'
# in non-conditional encodings, we want to remove the 8 one-hot cols, the two conditional cols, and the 'Label' col
# in conditional encodings, we also want to remove the two condition cols
stop_idx = -13 if conditional == 'conditional_' else -11


condition = 'TemperatureKelvin'
file_path = f'/home/cmdunham/ChemicalDataGeneration/data/encoder_embedding_predictions/{model_type}_{results_type.lower()}_preds.csv'
embedding_preds_df = pd.read_csv(file_path)

# add a col to embedding_preds_df called TemperatureKelvin. 
embedding_preds_df = ppf.merge_conditions(embedding_preds_df, metadata, col_to_insert_before='DEB')

#%%
# create a label column based on the one-hot encoded columns
one_hot_cols = embedding_preds_df.columns[-8:]
embedding_preds_df['Label'] = embedding_preds_df[one_hot_cols].idxmax(axis=1)

cols = list(embedding_preds_df.columns)
label_col = cols.pop(cols.index('Label'))
cols.insert(-8, label_col)
embedding_preds_df = embedding_preds_df[cols]
sorted_chem_names = ['DEB','DEM','DMMP','DPM','DtBP','JP8','MES','TEPO']
conditions = ['TemperatureKelvin', 'PressureBar']
encodings_list = embedding_preds_df[sorted_chem_names].values.tolist()
#%%
# Plot predicted TemperatureKelvin against predicted PressureBar
sorted_chem_names = ['DEB','DEM','DMMP','DPM','DtBP','JP8','MES','TEPO']

for chem in sorted_chem_names:
    chem_embeddings = embedding_preds_df[embedding_preds_df['Label'] == chem]
    plt.figure(figsize=(8, 6))
    plt.scatter(chem_embeddings['TemperatureKelvin'], chem_embeddings['PressureBar'], alpha=0.5)
    plt.xlabel('Temperature Kelvin', fontsize=16)
    plt.ylabel('Pressure Bar', fontsize=16)
    plt.title(f'True Temperature vs Pressure for {chem}', fontsize=20)
    # plt.xticks([])
    # plt.yticks([])
    plt.show()
#%%
importlib.reload(pf)

#%%
for condition in conditions:
    for chem in sorted_chem_names:
        chem_embeddings = embedding_preds_df[embedding_preds_df['Label'] == chem]
        if chem_embeddings.shape[0] < 2000:
            chem_embeddings = chem_embeddings.sample(frac=0.1, random_state=42)
        else:
            chem_embeddings = chem_embeddings.sample(frac=0.05, random_state=42)

        chemnet_embeddings_to_plot = ims_embeddings[[chem]]

        save_plot_path = f'../plots/CARL/encoder_results/{chem}_{conditional}embeddings_{plot_pred_condition}by_{condition}.png'
        plot_predicted_condition=True if plot_pred_condition == 'with_predicted_' else False
        pf.plot_emb_colored_by_condition(
            ims_embeddings, 
            chemnet_embeddings_to_plot, 
            chem_embeddings,
            chem,
            results_type,
            condition=condition,
            save_plot_path=save_plot_path,
            stop_idx=stop_idx,
            plot_predicted_condition=plot_predicted_condition,
            )
#%%
embeddings_only = embedding_preds_df.iloc[:,1:-11]
pf.plot_emb_pca(
    ims_embeddings, embeddings_only, results_type, 'IMS', 
    log_wandb=False, chemnet_embeddings_to_plot=chemnet_embeddings_to_plot,
    show_wandb_run_name=False)