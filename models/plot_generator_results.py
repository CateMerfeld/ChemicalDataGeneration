#%%
import pandas as pd
# import torch.nn as nn
import importlib
import functions as f
import plotting_functions as pf
# import random

#%%
importlib.reload(f)
#%%
# result_type = 'spectrum'
result_type = 'CARL'
model_type = 'universal_generator'
plot_type = 'real_vs_synthetic'
save_file_path_pt1 = f'../plots/{result_type}/generator_results/{model_type}/{plot_type}_'
save_file_path_pt2 = f'_{result_type}.png'

synthetic_data_file_ending = 'feather'
synthetic_data_path_pt_1 = f'../../scratch/synthetic_data/{result_type}/{model_type}/'
synthetic_data_path_pt_2 = f'synthetic_test_spectra.{synthetic_data_file_ending}'
test_file_path = '../../scratch/test_data.feather'
experimental_start_idx = 2
experimental_end_idx = -9
synthetic_end_idx = -1

sorted_chem_names = ['DEB','DEM','DMMP','DPM','DtBP','JP8','MES','TEPO']

if model_type == 'group_generator':
    chem_groups = [['DMMP', 'TEPO'], ['DEM', 'DPM', 'DEB'], ['DtBP', 'MES']]
else:
    chem_groups = None

experimental_data = pd.read_feather(test_file_path)

def load_data(file_path_parts_list, synthetic_data_file_ending):
    """
    Load data from a file path constructed from the provided parts.
    Args:
        file_path_parts_list (list): List of strings representing parts of the file path.   
        synthetic_data_file_ending (str): The file ending of the synthetic data file.
    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    file_path = '_'.join(file_path_parts_list)
    if synthetic_data_file_ending == 'feather':
        data = pd.read_feather(file_path)
    elif synthetic_data_file_ending == 'csv':
        data = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file ending: {synthetic_data_file_ending}")
    return data

if chem_groups is not None:
    for group in chem_groups:
        group_file_path = '_'.join(group)
        synthetic_data = load_data([synthetic_data_path_pt_1, group_file_path, synthetic_data_path_pt_2], synthetic_data_file_ending)

        for chem in group:
            print(f'Plotting {chem}...')
            experimental_chem_data = experimental_data[experimental_data['Label'] == chem].iloc[:,experimental_start_idx:experimental_end_idx]
            synthetic_chem_data = synthetic_data[synthetic_data['Label'] == chem].iloc[:,:synthetic_end_idx]
            pf.plot_average_spectrum(
                experimental_chem_data, synthetic_chem_data, 
                chem_label=chem, 
                save_file_path_pt1=save_file_path_pt1, 
                save_file_path_pt2=save_file_path_pt2,
                plot_1_type='Experimental ', plot_2_type='Synthetic '
                )
else:
    synthetic_data = load_data([synthetic_data_path_pt_1,synthetic_data_path_pt_2], synthetic_data_file_ending)
    
    for chem in sorted_chem_names:
        print(f'Plotting {chem}...')
        experimental_chem_data = experimental_data[experimental_data['Label'] == chem].iloc[:,experimental_start_idx:experimental_end_idx]
        synthetic_chem_data = synthetic_data[synthetic_data['Label'] == chem].iloc[:,:synthetic_end_idx]
        pf.plot_average_spectrum(
            experimental_chem_data, synthetic_chem_data, 
            chem_label=chem, 
            save_file_path_pt1=save_file_path_pt1, 
            save_file_path_pt2=save_file_path_pt2,
            plot_1_type='Experimental ', plot_2_type='Synthetic '
            )
###############################################

# # # for group generators
# # for group in chem_groups:
# #     group_file_path = '_'.join(group)
# #     synthetic_data_path = '_'.join([synthetic_data_path_pt_1,group_file_path,synthetic_data_path_pt_2])
# #     synthetic_data = pd.read_csv(synthetic_data_path)

# #     for chem in group:
# # synthetic_data = pd.read_csv(synthetic_data_path)

# # for universal generator
# synthetic_data_path = '_'.join([synthetic_data_path_pt_1,synthetic_data_path_pt_2])
# if synthetic_data_file_ending == 'feather':
#     synthetic_data = pd.read_feather(synthetic_data_path)
# elif synthetic_data_file_ending == 'csv':
#     synthetic_data = pd.read_csv(synthetic_data_path)

# for chem in sorted_chem_names:
#     print(f'Plotting {chem}...')
#     experimental_chem_data = experimental_data[experimental_data['Label'] == chem].iloc[:,2:-9]
#     synthetic_chem_data = synthetic_data[synthetic_data['Label'] == chem].iloc[:,:-2]

#     pf.plot_average_spectrum(
#         experimental_chem_data, synthetic_chem_data, 
#         chem_label=chem, 
#         save_file_path_pt1=save_file_path_pt1, 
#         save_file_path_pt2=save_file_path_pt2,
#         plot_1_type='Experimental ', plot_2_type='Synthetic '
#         )


#%%

# file_path = '../../scratch/test_data.feather'
# spectra = pd.read_feather(file_path)
# #%%
# chem = 'MES'
# # gen_type = 'Individual'
# file_path = f'../data/ims_data/synthetic_test_{chem}_spectra.csv'
# synthetic_spectra_df = pd.read_csv(file_path)
# file_path = f'../data/ims_data/synthetic_test_{chem}_spectra_universal_generator.csv'
# synthetic_spectra_df_universal = pd.read_csv(file_path)

# #%%
# results_type ='Test'
# num_plots = 5
# criterion = nn.MSELoss()
# test_spectra_single_chem = spectra[spectra['Label'] == chem]
# indices = list(test_spectra_single_chem['index'])
# file_path = '../plots/generator_results/'
# #%%
# input_spectra, sorted_chem_names, input_labels = f.format_data_for_plotting(spectra)
# synthetic_spectra, synthetic_spectra_labels = f.format_data_for_plotting(synthetic_spectra_df)
# #%%
# f.plot_generation_results_pca_single_chem_side_by_side(
#     input_spectra, synthetic_spectra, sorted_chem_names, 
#     results_type=results_type, chem_of_interest=chem, save_plot_path=file_path
#     )
# #%%
# for i in range(num_plots):
#     random_spec_idx = random.choice(indices)
#     true_spec = test_spectra_single_chem[test_spectra_single_chem['index']==random_spec_idx]
#     synthetic_spec = synthetic_spectra_df[synthetic_spectra_df['index']==random_spec_idx]
#     true_spec_values = true_spec.iloc[:, 2:-9].values.flatten()
#     synthetic_spec_values = synthetic_spec.iloc[:, 2:-1].values.flatten()
#     file_path = f'../plots/generator_results/{chem}_real_synthetic_comparison_{i}.png'
#     f.plot_carl_real_synthetic_comparison(
#         true_spec_values, synthetic_spec_values, results_type, 
#         chem, criterion=criterion, save_plot_path=file_path
#         )
# #%
# #%%
# pf.plot_similarity_comparison(spectra, chem, synthetic_spectra_df, 'Individual', 2, -1)#, similarity_type='spect_avg')
# pf.plot_similarity_comparison(spectra, chem, synthetic_spectra_df_universal, 'Universal', 0, -1)#, similarity_type='spect_avg')
#%%