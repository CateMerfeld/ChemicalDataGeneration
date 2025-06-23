#%%
import functions as f
import plotting_functions as pf
import pandas as pd
name_smiles_embedding_df_file_path = '../../scratch/name_smiles_embedding_file.csv'
name_smiles_embedding_df = f.format_embedding_df(name_smiles_embedding_df_file_path)

ims_embeddings = pd.DataFrame([emb for emb in name_smiles_embedding_df['Embedding Floats']][1:]).T
cols = name_smiles_embedding_df.index[1:]
ims_embeddings.columns = cols
#%%
results_type = 'Train'
file_path = '../data/encoder_embedding_predictions/high_temp_conditioning_train_preds.csv'
embedding_preds_df = pd.read_csv(file_path)
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
embeddings_only = embedding_preds_df.iloc[:,1:-8]

pf.plot_emb_pca(
    ims_embeddings, embeddings_only, results_type, 'IMS', 
    log_wandb=False, chemnet_embeddings_to_plot=ims_embeddings,
    show_wandb_run_name=False)