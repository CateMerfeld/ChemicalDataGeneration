#%%
import pandas as pd
import torch.nn as nn
import importlib
import functions as f
import random
#%%
importlib.reload(f)
#%%
file_path = '../../scratch/test_data.feather'
spectra = pd.read_feather(file_path)
file_path = '../data/ims_data/synthetic_test_spectra.csv'
synthetic_spectra_df = pd.read_csv(file_path)
#%%
input_spectra, sorted_chem_names, input_labels = f.format_data_for_plotting(spectra)
synthetic_spectra, synthetic_spectra_labels = f.format_data_for_plotting(synthetic_spectra_df)
#%%
chem = 'JP8'
results_type ='Test'
file_path = '../plots/generator_results/'
f.plot_generation_results_pca_single_chem_side_by_side(
    input_spectra, synthetic_spectra, sorted_chem_names, 
    results_type=results_type, chem_of_interest=chem, save_plot_path=file_path
    )
#%%
num_plots = 5
chem = 'JP8'
criterion = nn.MSELoss()
results_type ='Test'
test_spectra_single_chem = spectra[spectra['Label'] == chem]
indices = list(test_spectra_single_chem['index'])
file_path = '../plots/generator_results/'

for i in range(num_plots):
    random_spec_idx = random.choice(indices)
    true_spec = test_spectra_single_chem[test_spectra_single_chem['index']==random_spec_idx]
    synthetic_spec = synthetic_spectra_df[synthetic_spectra_df['index']==random_spec_idx]
    true_spec_values = true_spec.iloc[:, 2:-9].values.flatten()
    synthetic_spec_values = synthetic_spec.iloc[:, 2:-1].values.flatten()
    file_path = f'../plots/generator_results/{chem}_real_synthetic_comparison_{i}.png'
    f.plot_carl_real_synthetic_comparison(
        true_spec_values, synthetic_spec_values, results_type, 
        chem, criterion=criterion, save_plot_path=file_path
        )
#%%