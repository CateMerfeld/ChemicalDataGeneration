#%%
import pandas as pd
# import torch.nn as nn
import functions as f
import plotting_functions as pf
# import random
#%%
import importlib
#%%
importlib.reload(pf)
#%%
chem_label = 'DMMP'
save_plot_path=f'../plots/CARL/generator_results/group_generator/{chem_label}_experimental_vs_synthetic_multiple_spectra_same_plot.png'
#%%
synthetic_data_path = '../../scratch/synthetic_data/CARL/group_generator/_DMMP_TEPO_synthetic_test_spectra.csv'
synthetic_data = pd.read_csv(synthetic_data_path)
#%%
# synthetic_phil_path = 

#%%
experimental_data_path = '../../scratch/test_data.feather'
experimental_data = pd.read_feather(experimental_data_path)
#%%
chem_synthetic_data = synthetic_data[synthetic_data['Label'] == chem_label].iloc[:,:-2]
chem_synthetic_data = chem_synthetic_data.sample(n=30, random_state=10, ignore_index=True)

print(chem_synthetic_data.head(10))
#%%
chem_experimental_data = experimental_data[experimental_data['Label'] == chem_label].iloc[:,2:-9]
chem_experimental_data = chem_experimental_data.sample(n=30, random_state=42, ignore_index=True)
#%%
importlib.reload(pf)
pf.plot_comparison_multiple_spectra_per_plot(
    chem_experimental_data, chem_synthetic_data,
    dataset1_type='Experimental Test Spectra', dataset2_type='Synthetic Test Spectra from CARLs',
    chem_label=chem_label, save_plot_path=save_plot_path
    )

#%%
# importlib.reload(f)
# #%%
# middle_plot_result_type = 'PHIL'
# right_plot_result_type = 'CARL'
# left_plot_result_type = 'experimental'
# # model type options: 'group_generator', 'universal_generator', 'individual_generators'
# model_type = 'universal_generator'
# # plot_type = 'real_vs_synthetic'
# result_type = 'PHIL'
# plot_type = 'model_comparison'

# if plot_type == 'model_comparison':
#     save_file_path_pt1 = f'../plots/{plot_type}/'
#     save_file_path_pt2 = f'_{middle_plot_result_type}_{right_plot_result_type}_{model_type}.png'
# else:
#     save_file_path_pt1 = f'../plots/{result_type}/generator_results/{model_type}/{plot_type}_'
#     save_file_path_pt2 = f'_{result_type}.png'


# middle_plot_data_file_ending = 'feather'   
# right_plot_data_file_ending = 'feather'

# test_file_path = '../../scratch/test_data.feather'
# left_plot_start_idx = 2
# left_plot_stop_idx = -9

# middle_plot_start_idx = 0
# middle_plot_stop_idx = -2
# right_plot_stop_idx = -1

# left_plot_type = 'Experimental '
# middle_plot_type = 'PHIL '
# right_plot_type = 'CARL '
# data_split = 'Spectrum vs CARL'

# # Everything below this line should not need to be changed
# ##########################################################

# middle_plot_data_path_pt_1 = f'../../scratch/synthetic_data/{middle_plot_result_type}/{model_type}/'
# middle_plot_data_path_pt_2 = f'synthetic_test_spectra.{middle_plot_data_file_ending}'

# right_plot_data_path_pt_1 = f'../../scratch/synthetic_data/{right_plot_result_type}/{model_type}/'
# right_plot_data_path_pt_2 = f'synthetic_test_spectra.{right_plot_data_file_ending}'

# sorted_chem_names = ['DEB','DEM','DMMP','DPM','DtBP','JP8','MES','TEPO']

# if model_type == 'group_generator':
#     chem_groups = [['DMMP', 'TEPO']]#, ['DEM', 'DPM', 'DEB'], ['DtBP', 'MES']]
# else:
#     chem_groups = None

# left_plot_data = pd.read_feather(test_file_path)

# # left_plot_data = experimental_data.iloc[:,experimental_start_idx:experimental_end_idx]


# # middle_plot_data = f.load_data([middle_plot_data_path_pt_1, middle_plot_data_path_pt_2], middle_plot_data_file_ending)

# right_plot_data = f.load_data([right_plot_data_path_pt_1, right_plot_data_path_pt_2], right_plot_data_file_ending)
# for chem in sorted_chem_names:
#     save_plot_path = f'../plots/{plot_type}/{chem}_{left_plot_result_type}_vs_{middle_plot_result_type}_{model_type}_pca.png'
#     pf.plot_generation_results_pca_single_chem_side_by_side(
#         left_plot_data, right_plot_data, sorted_chem_names, results_type=right_plot_type,
#         sample_size=1000, chem_of_interest=chem, save_plot_path=save_plot_path,
#         true_spectra_start_idx=left_plot_start_idx, true_spectra_stop_idx=left_plot_stop_idx, 
#         synthetic_spectra_stop_idx=middle_plot_stop_idx,
#         )
#     # save_plot_path = f'../plots/{plot_type}/{chem}_{left_plot_result_type}_vs_{right_plot_result_type}_{model_type}_pca.png'
#     # pf.plot_generation_results_pca_single_chem_side_by_side(
#     #     left_plot_data, right_plot_data, sorted_chem_names, results_type=right_plot_type,
#     #     sample_size=1000, chem_of_interest=chem, save_plot_path=save_plot_path,
#     #     true_spectra_start_idx=left_plot_start_idx, true_spectra_stop_idx=left_plot_stop_idx, 
#     #     synthetic_spectra_stop_idx=right_plot_stop_idx,
#     #     )

# # if chem_groups is not None:
# #     for group in chem_groups:
# #         group_file_path = '_'.join(group)
# #         middle_plot_data = f.load_data([middle_plot_data_path_pt_1, group_file_path, middle_plot_data_path_pt_2], middle_plot_data_file_ending)
# #         right_plot_data = f.load_data([right_plot_data_path_pt_1, group_file_path, right_plot_data_path_pt_2], right_plot_data_file_ending)

# #         pf.plot_average_spectrum(
# #             left_plot_data, middle_plot_data, chem_names=group, 
# #             save_file_path_pt1=save_file_path_pt1, save_file_path_pt2=save_file_path_pt2,
# #             left_plot_type=left_plot_type, middle_plot_type=middle_plot_type,
# #             right_plot_data=right_plot_data, right_plot_type=right_plot_type, condition_3_value='',
# #             left_plot_start_idx=left_plot_start_idx, left_plot_stop_idx=left_plot_stop_idx,
# #             middle_plot_stop_idx=middle_plot_stop_idx, right_plot_stop_idx=right_plot_stop_idx,
# #             )
        
# #         # pf.plot_conditions_pca(
# #         #     left_plot_data, right_plot_data, save_file_path_pt1, save_file_path_pt2,
# #         #     data_one_name=left_plot_type, data_two_name=right_plot_type, data_split=data_split,
# #         #     fit_to_all=False
# #         #     )
      
# # else:
# #     middle_plot_data = f.load_data([middle_plot_data_path_pt_1, middle_plot_data_path_pt_2], middle_plot_data_file_ending)
# #     right_plot_data = f.load_data([right_plot_data_path_pt_1, right_plot_data_path_pt_2], right_plot_data_file_ending)
# #     pf.plot_average_spectrum(
# #         left_plot_data, middle_plot_data, chem_names=sorted_chem_names, 
# #         save_file_path_pt1=save_file_path_pt1, save_file_path_pt2=save_file_path_pt2,
# #         left_plot_type=left_plot_type, middle_plot_type=middle_plot_type,
# #         right_plot_data=right_plot_data, right_plot_type=right_plot_type, condition_3_value='',
# #         left_plot_start_idx=left_plot_start_idx, left_plot_stop_idx=left_plot_stop_idx,
# #         middle_plot_stop_idx=middle_plot_stop_idx, right_plot_stop_idx=right_plot_stop_idx,
# #         )
#     # print(middle_plot_data.head())
#     # save_file_path_pt1 = f'../plots/{result_type}/generator_results/{model_type}/{plot_type}_'
#     # save_file_path_pt2 = f'_{result_type}_pca.png'

#     # save_file_path_pt1 = f'../plots/{plot_type}/'
#     # save_file_path_pt2 = f'_experimental_{middle_plot_result_type}_{model_type}_pca.png'

#     # pf.plot_generation_results_pca()
#     # pf.plot_generation_results_pca_single_chem_side_by_side()

#     # pf.plot_conditions_pca(
#     #         left_plot_data, middle_plot_data, save_file_path_pt1, save_file_path_pt2,
#     #         data_one_name=left_plot_type, data_two_name=middle_plot_type, data_split=data_split,
#     #         fit_to_all=False, condition_two_start_idx=middle_plot_start_idx, condition_two_stop_idx=middle_plot_stop_idx,
#     #         )



# ###############################################

# # # # for group generators
# # # for group in chem_groups:
# # #     group_file_path = '_'.join(group)
# # #     synthetic_data_path = '_'.join([synthetic_data_path_pt_1,group_file_path,synthetic_data_path_pt_2])
# # #     synthetic_data = pd.read_csv(synthetic_data_path)

# # #     for chem in group:
# # # synthetic_data = pd.read_csv(synthetic_data_path)

# # # for universal generator
# # synthetic_data_path = '_'.join([synthetic_data_path_pt_1,synthetic_data_path_pt_2])
# # if synthetic_data_file_ending == 'feather':
# #     synthetic_data = pd.read_feather(synthetic_data_path)
# # elif synthetic_data_file_ending == 'csv':
# #     synthetic_data = pd.read_csv(synthetic_data_path)

# # for chem in sorted_chem_names:
# #     print(f'Plotting {chem}...')
# #     experimental_chem_data = experimental_data[experimental_data['Label'] == chem].iloc[:,2:-9]
# #     synthetic_chem_data = synthetic_data[synthetic_data['Label'] == chem].iloc[:,:-2]

# #     pf.plot_average_spectrum(
# #         experimental_chem_data, synthetic_chem_data, 
# #         chem_label=chem, 
# #         save_file_path_pt1=save_file_path_pt1, 
# #         save_file_path_pt2=save_file_path_pt2,
# #         plot_1_type='Experimental ', plot_2_type='Synthetic '
# #         )


# #%%

# # file_path = '../../scratch/test_data.feather'
# # spectra = pd.read_feather(file_path)
# # #%%
# # chem = 'MES'
# # # gen_type = 'Individual'
# # file_path = f'../data/ims_data/synthetic_test_{chem}_spectra.csv'
# # synthetic_spectra_df = pd.read_csv(file_path)
# # file_path = f'../data/ims_data/synthetic_test_{chem}_spectra_universal_generator.csv'
# # synthetic_spectra_df_universal = pd.read_csv(file_path)

# # #%%
# # results_type ='Test'
# # num_plots = 5
# # criterion = nn.MSELoss()
# # test_spectra_single_chem = spectra[spectra['Label'] == chem]
# # indices = list(test_spectra_single_chem['index'])
# # file_path = '../plots/generator_results/'
# # #%%
# # input_spectra, sorted_chem_names, input_labels = f.format_data_for_plotting(spectra)
# # synthetic_spectra, synthetic_spectra_labels = f.format_data_for_plotting(synthetic_spectra_df)
# # #%%
# # f.plot_generation_results_pca_single_chem_side_by_side(
# #     input_spectra, synthetic_spectra, sorted_chem_names, 
# #     results_type=results_type, chem_of_interest=chem, save_plot_path=file_path
# #     )
# # #%%
# # for i in range(num_plots):
# #     random_spec_idx = random.choice(indices)
# #     true_spec = test_spectra_single_chem[test_spectra_single_chem['index']==random_spec_idx]
# #     synthetic_spec = synthetic_spectra_df[synthetic_spectra_df['index']==random_spec_idx]
# #     true_spec_values = true_spec.iloc[:, 2:-9].values.flatten()
# #     synthetic_spec_values = synthetic_spec.iloc[:, 2:-1].values.flatten()
# #     file_path = f'../plots/generator_results/{chem}_real_synthetic_comparison_{i}.png'
# #     f.plot_carl_real_synthetic_comparison(
# #         true_spec_values, synthetic_spec_values, results_type, 
# #         chem, criterion=criterion, save_plot_path=file_path
# #         )
# # #%
# # #%%
# # pf.plot_similarity_comparison(spectra, chem, synthetic_spectra_df, 'Individual', 2, -1)#, similarity_type='spect_avg')
# # pf.plot_similarity_comparison(spectra, chem, synthetic_spectra_df_universal, 'Universal', 0, -1)#, similarity_type='spect_avg')
# #%%