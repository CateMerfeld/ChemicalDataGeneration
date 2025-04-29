#%%
import functions as f
import pandas as pd
name_smiles_embedding_df_file_path = '../../scratch/name_smiles_embedding_file.csv'
name_smiles_embedding_df = f.format_embedding_df(name_smiles_embedding_df_file_path)

ims_embeddings = pd.DataFrame([emb for emb in name_smiles_embedding_df['Embedding Floats']][1:]).T
cols = name_smiles_embedding_df.index[1:]
ims_embeddings.columns = cols
#%%
file_path = '../data/encoder_embedding_predictions/reparameterization_test_preds.csv'
test_preds_df = pd.read_csv(file_path)
sorted_chem_names = ['DEB','DEM','DMMP','DPM','DtBP','JP8','MES','TEPO']
encodings_list = test_preds_df[sorted_chem_names].values.tolist()
# spectra_labels = [sorted_chem_names[list(enc).index(1)] for enc in encodings_list]
embeddings_only = test_preds_df.iloc[:,1:-8]

f.plot_emb_pca(
    ims_embeddings, embeddings_only, 'Test', 'IMS', 
    log_wandb=False, chemnet_embeddings_to_plot=ims_embeddings,
    show_wandb_run_name=False)