#%%
import pandas as pd
# #%%
# # import importlib
import time
#%%
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
# %%
import plotting_functions as pf
# import functions as f
# importlib.reload(pf)
# #%%

input_condition_value = 'High'
output_condition_value = 'Low'
condition_name = 'Temperature'
synthetic_condition_value='Low'
file_path_condition_name = 'TemperatureKelvin' 
model_type = f'universal_{output_condition_value.lower()}_{condition_name}_generator'
# #%%
print(f'Loading {input_condition_value} {condition_name} experimental data...')
start_time = time.time()
file_path = f'../../../scratch/train_spectra_{input_condition_value.lower()}_TemperatureKelvin.csv'
encoder_train_data = pd.read_csv(file_path)
encoder_train_data = encoder_train_data.sample(frac=.1, random_state=42)
end_time = time.time()
print(f'Loaded in {round(end_time - start_time, 3)} seconds')
#%%
print(f'Loading {output_condition_value} {condition_name} experimental data...')
start_time = time.time()
file_path = f'../../../scratch/train_spectra_{output_condition_value.lower()}_TemperatureKelvin.csv'
gen_train_data = pd.read_csv(file_path)
gen_train_data = gen_train_data.sample(frac=.8, random_state=42)
end_time = time.time()
print(f'Loaded in {round(end_time - start_time, 3)} seconds')

plotting_chem = 'DPM'
train_test_val='Train'
save_plot_path = f'../../plots/{plotting_chem}_{input_condition_value}_vs_{output_condition_value}_{condition_name}_.png'
pf.plot_carl_real_synthetic_comparison(
    true_carl=encoder_train_data[encoder_train_data['Label'] == plotting_chem].iloc[0, 2:-9], 
    synthetic_carl=gen_train_data[gen_train_data['Label'] == plotting_chem].iloc[0, 2:-9],
    results_type=train_test_val,
    chem_label=plotting_chem,
    save_plot_path=save_plot_path,
    carl_or_spec='Spectrum',
    comparison_names=[f'{input_condition_value} {condition_name}', f'{output_condition_value} {condition_name}'],
    )

# print(f'Loading {output_condition_value} {condition_name} synthetic data...')
# start_time = time.time()
# file_path = f'../../../scratch/synthetic_data/synthetic_test_low_temp___universal_low_temp_generator_spectra.csv'

# preds = pd.read_csv(file_path)
# preds = preds.sample(frac=.8, random_state=42)
# end_time = time.time()
# print(f'Loaded in {round(end_time - start_time, 3)} seconds')

# # print(preds.shape)
# # print(preds.head())

# # #%%
# plot_type = 'real_vs_synthetic'
# save_file_path_pt1 = f'../../plots/avg_high_vs_low_{condition_name}_{plot_type}_'
# save_file_path_pt2 = '_spectra.png'
# plot_title_condition = 'Temperature'

# for chem in gen_train_data['Label'].unique():
#     if chem in encoder_train_data['Label'].unique():
#         print(f'Plotting {chem}...')
#         low_temp_chem = encoder_train_data[encoder_train_data['Label'] == chem].iloc[:,2:-9]
#         high_temp_chem = gen_train_data[gen_train_data['Label'] == chem].iloc[:,2:-9]
#         high_temp_chem_preds = preds[preds['Label'] == chem].iloc[:,:-1]

#         pf.plot_average_spectrum(
#             low_temp_chem, high_temp_chem, 
#             chem_label=chem, 
#             condition_name=condition_name,
#             condition_1_value=input_condition_value,
#             condition_2_value=output_condition_value,
#             save_file_path_pt1=save_file_path_pt1, 
#             save_file_path_pt2=save_file_path_pt2,
#             synthetic_data=high_temp_chem_preds,
#             synthetic_condition_type=synthetic_condition_value
#             )