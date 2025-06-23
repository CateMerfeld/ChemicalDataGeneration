#%%
import functions as f
import plotting_functions as pf
import pandas as pd
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data_preprocessing'))
sys.path.append(parent_dir)
import preprocessing_functions as ppf
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

metadata = pd.read_feather('../../scratch/BKG_SIM_ims_acbc_train_v1.1.09_meta.feather')

name_smiles_embedding_df_file_path = '../../scratch/name_smiles_embedding_file.csv'
name_smiles_embedding_df = f.format_embedding_df(name_smiles_embedding_df_file_path)

ims_embeddings = pd.DataFrame([emb for emb in name_smiles_embedding_df['Embedding Floats']][1:]).T
cols = name_smiles_embedding_df.index[1:]
ims_embeddings.columns = cols
#%%
results_type = 'Test'
file_path = f'../data/encoder_embedding_predictions/carl_to_chemnet_{results_type.lower()}_preds.csv'
embedding_preds_df = pd.read_csv(file_path)

# add a col to embedding_preds_df called TemperatureKelvin. 
embedding_preds_df = ppf.merge_conditions(embedding_preds_df, metadata, col_to_insert_before='DEB')
print(embedding_preds_df.head())
#%%
# create a label column based on the one-hot encoded columns
one_hot_cols = embedding_preds_df.columns[-8:]
embedding_preds_df['Label'] = embedding_preds_df[one_hot_cols].idxmax(axis=1)

cols = list(embedding_preds_df.columns)
label_col = cols.pop(cols.index('Label'))
cols.insert(-8, label_col)
embedding_preds_df = embedding_preds_df[cols]
#%%
sorted_chem_names = ['DEB','DEM','DMMP','DPM','DtBP','JP8','MES','TEPO']
encodings_list = embedding_preds_df[sorted_chem_names].values.tolist()

# spectra_labels = [sorted_chem_names[list(enc).index(1)] for enc in encodings_list]
embeddings_only = embedding_preds_df.iloc[:,1:-11]
#%%
chem = 'DtBP'
chem_embeddings = embedding_preds_df[embedding_preds_df['Label'] == chem]
chemnet_embeddings_to_plot = ims_embeddings[[chem]]
#%%

# Fit PCA on ims_embeddings
pca = PCA(n_components=2)
ims_pca = pca.fit(ims_embeddings.T)

# Transform chemnet_embeddings_to_plot
chemnet_pca = pca.transform(chemnet_embeddings_to_plot.T)

# Transform embeddings_only
embeddings_only = chem_embeddings.iloc[:,1:-11]
embeddings_only_pca = pca.transform(embeddings_only)

# Plot chemnet_embeddings_to_plot
plt.scatter(chemnet_pca[:, 0], chemnet_pca[:, 1], color='black', label=f'ChemNet Embeddings ({chem})', s=200, alpha=0.6)

# Plot embeddings_only, colored by TemperatureKelvin
temp = chem_embeddings['TemperatureKelvin']
sc = plt.scatter(embeddings_only_pca[:, 0], embeddings_only_pca[:, 1], c=temp, marker='x', cmap='viridis', label='Predicted Embeddings')
plt.colorbar(sc, label='Temperature (K)')

# plt.xlabel('PCA 1')
# plt.ylabel('PCA 2')
plt.xticks([])
plt.yticks([])
plt.legend()
plt.title(f'PCA of {chem} Embeddings')
plt.show()
#%%
pf.plot_emb_pca(
    ims_embeddings, embeddings_only, results_type, 'IMS', 
    log_wandb=False, chemnet_embeddings_to_plot=chemnet_embeddings_to_plot,
    show_wandb_run_name=False)