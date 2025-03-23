#%%
import pandas as pd
meta = pd.read_feather('../../data/BKG_SIM_ims_acbc_train_v1.1.09_meta.feather')
print(meta.head())
# #%%
# # import importlib
# import time
# #%%
# import os
# import sys
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.append(parent_dir)
# # %%
# import plotting_functions as pf
# # import functions as f
# # importlib.reload(pf)
# # #%%
# condition = 'TemperatureKelvin'
# synthetic_condition_type='High'
# # # #%%
# print(f'Loading low {condition} experimental data...')
# start_time = time.time()
# file_path = '../../../scratch/train_spectra_low_TemperatureKelvin.csv'
# train_low_cond = pd.read_csv(file_path)
# train_low_cond = train_low_cond.sample(frac=.1, random_state=42)
# end_time = time.time()
# print(f'Loaded in {round(end_time - start_time, 3)} seconds')
# #%%
# print(f'Loading high {condition} experimental data...')
# start_time = time.time()
# file_path = '../../../scratch/train_spectra_high_TemperatureKelvin.csv'
# train_high_cond = pd.read_csv(file_path)
# train_high_cond = train_high_cond.sample(frac=.8, random_state=42)
# end_time = time.time()
# print(f'Loaded in {round(end_time - start_time, 3)} seconds')
# # dmmp = train_high_cond[train_high_cond['Label'] == 'DMMP']
# # print(dmmp.head())


# # print(f'Loading high {condition} embedding_preds...')
# # start_time = time.time()
# # file_path = '../../data/encoder_embedding_predictions/conditioning_test_preds.csv'
# # train_embeddings_df = pd.read_csv(file_path)
# # end_time = time.time()
# # print(f'Loaded in {round(end_time - start_time, 3)} seconds')
# # print(train_embeddings_df.head())
# print(f'Loading {synthetic_condition_type} {condition} synthetic data...')
# start_time = time.time()
# # file_path = f'../../data/ims_data/synthetic_test_high_TemperatureKelvin_all_spectra_universal_generator_spectra.csv'
# file_path = f'../../../scratch/synthetic_data/synthetic_test_high_TemperatureKelvin_all_chemicals_universal_generator_spectra.csv'
# # file_path = f'../../../scratch/synthetic_data/synthetic_test_high_TemperatureKelvin_dmmp_tepo_on_universal_generator_spectra.csv'
# # file_path = f'../../data/ims_data/synthetic_test_high_TemperatureKelvin_all_chemicals_universal_generator_undertrained_spectra.csv'
# preds = pd.read_csv(file_path)
# # preds = preds.sample(frac=.8, random_state=42)
# end_time = time.time()
# print(f'Loaded in {round(end_time - start_time, 3)} seconds')

# # #%%
# # # # # importlib.reload(pf)
# # plot_type = 'real_vs_synthetic_dmmp_tepo'
# plot_type = 'real_vs_synthetic'
# save_file_path_pt1 = f'../../plots/avg_low_vs_high_{condition}_{plot_type}_'
# save_file_path_pt2 = '_spectra.png'
# plot_title_condition = 'Temperature'

# for chem in train_high_cond['Label'].unique():
#     print(f'Plotting {chem}...')
#     low_temp_chem = train_low_cond[train_low_cond['Label'] == chem].iloc[:,2:-9]
#     high_temp_chem = train_high_cond[train_high_cond['Label'] == chem].iloc[:,2:-9]
#     high_temp_chem_preds = preds[preds['Label'] == chem].iloc[:,:-1]

#     pf.plot_average_spectrum(
#         low_temp_chem,
#         high_temp_chem,  
#         chem, 
#         save_file_path_pt1=save_file_path_pt1, 
#         save_file_path_pt2=save_file_path_pt2,
#         condition=plot_title_condition,
#         synthetic_data=high_temp_chem_preds,
#         synthetic_condition_type=synthetic_condition_type
#         )