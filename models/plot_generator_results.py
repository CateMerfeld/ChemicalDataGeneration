#%%
import pandas as pd
import torch.nn as nn
import importlib
import functions as f
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial import distance
#%%
importlib.reload(f)
#%%
# file_path = '../../scratch/test_data.feather'
# spectra = pd.read_feather(file_path)
chem = 'JP8'
# gen_type = 'Individual'
file_path = f'../data/ims_data/synthetic_test_{chem}_spectra.csv'
synthetic_spectra_df = pd.read_csv(file_path)
file_path = f'../data/ims_data/synthetic_test_{chem}_spectra_universal_generator.csv'
synthetic_spectra_df_universal = pd.read_csv(file_path)
#%%
results_type ='Test'
num_plots = 5
criterion = nn.MSELoss()
test_spectra_single_chem = spectra[spectra['Label'] == chem]
indices = list(test_spectra_single_chem['index'])
file_path = '../plots/generator_results/'
#%%
input_spectra, sorted_chem_names, input_labels = f.format_data_for_plotting(spectra)
synthetic_spectra, synthetic_spectra_labels = f.format_data_for_plotting(synthetic_spectra_df)
#%%
f.plot_generation_results_pca_single_chem_side_by_side(
    input_spectra, synthetic_spectra, sorted_chem_names, 
    results_type=results_type, chem_of_interest=chem, save_plot_path=file_path
    )
#%%
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
# from itertools import combinations
# import numpy as np
# #%%
# average_sims = {}
# sims = {}
# criterion = nn.MSELoss()
# # Calculate pairwise similarity between all spectra for each chemical
# for chem in sorted_chem_names:
#     chem_similarities = []
#     chem_subset = spectra.loc[spectra['Label'] == chem]
#     chem_subset = chem_subset.drop(columns='Label').T
#     for i, j in combinations(chem_subset.columns, 2):
#         similarity = criterion(chem_subset[i], chem_subset[j])
#         chem_similarities.append(round(similarity, 2))
    
#     sims[chem] = chem_similarities
#     avg_sim = np.mean(chem_similarities)
#     average_sims[chem] = avg_sim
#%%
# chem = 'MES'
# Calculate pairwise similarity between all spectra for a chemical
# Only using a subset of the data to calculate similarity as it requires a lot of memory
chem_spectra = spectra[spectra['Label'] == chem].iloc[:, 2:-9]

chem_subset = chem_spectra.sample(n=100, random_state=42)
chem_subset_list = np.array(chem_subset.values.tolist())
mse_matrix = distance.cdist(chem_subset_list, chem_subset_list, 'euclidean')

# only using upper triangular part of matrix as all other values are duplicates
upper_triangular_mse = mse_matrix[np.triu_indices(mse_matrix.shape[0], k=1)]

average_difference = np.mean(upper_triangular_mse)

#%%
# Calculate pairwise similarity between all synthetic spectra and all real spectra for a chemical
chem_subset_synthetic = synthetic_spectra_df.sample(n=100, random_state=42).iloc[:, 2:-1]

chem_subset_synthetic_list = np.array(chem_subset_synthetic.values.tolist())
# mse_matrix_real_synthetic = np.mean((chem_subset_list[:, np.newaxis] - chem_subset_synthetic_list[np.newaxis, :]) ** 2, axis=2).flatten()
mse_matrix_real_synthetic = distance.cdist(chem_subset_list, chem_subset_synthetic_list, 'euclidean').flatten()
# divide each element of mse_matrix_real_synthetic by average_distance
normalized_mse_matrix_real_synthetic = mse_matrix_real_synthetic / average_difference
normalized_mse_matrix_real = mse_matrix.flatten() / average_difference
#%%
# Calculate pairwise similarity between all synthetic spectra and all real spectra for a chemical
chem_subset_synthetic_universal = synthetic_spectra_df_universal.sample(n=100, random_state=42).iloc[:, :-1]

chem_subset_synthetic_list_universal = np.array(chem_subset_synthetic_universal.values.tolist())
# mse_matrix_real_synthetic = np.mean((chem_subset_list[:, np.newaxis] - chem_subset_synthetic_list[np.newaxis, :]) ** 2, axis=2).flatten()
mse_matrix_real_synthetic_universal = distance.cdist(chem_subset_list, chem_subset_synthetic_list_universal, 'euclidean').flatten()
# divide each element of mse_matrix_real_synthetic by average_distance
normalized_mse_matrix_real_synthetic_universal = mse_matrix_real_synthetic_universal / average_difference

#%%
sns.histplot(normalized_mse_matrix_real_synthetic, bins=10, kde=False, color='darkgreen', alpha=0.5, label='Individual')
sns.histplot(normalized_mse_matrix_real_synthetic_universal, bins=10, kde=False, color='blue', alpha=0.5, label='Universal')
# sns.histplot(normalized_mse_matrix_real, bins=10, kde=False, color='blue', alpha=0.5, label='Real')

# Set the labels and title
plt.xlabel('Similarity', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.legend()
# plt.title(f'{chem} Normalized Similarity {gen_type} Gen.', fontsize=16)
plt.title(f'{chem} Normalized Similarity', fontsize=16)

plt.show()
#%%
# thing1 = [[1,2],[3,4,5]]
# thing2 = 2
# thing3 = thing1 / thing2
# print(thing3)

# print(average_difference)
#%%
# synthetic_spectra_df.head()
# print(off_diagonal_mse)
# print(off_diagonal_mse.flatten())
# print(upper_triangular_mse)
#%%
list_a = np.array([[0.1, 0.2, 0.3],  # Element 1
                #    [0.4, 0.5, 0.6],  # Element 2
                #    [0.7, 0.8, 0.9]
                    ]) # Element 3

list_b = np.array([[3, 0.5, 0.3],  # Element 1
                #    [2, 0.7, 0.6],  # Element 2
                #    [1, 0.6, 0.9]
                ]) # Element 3

#%%
# Calculate MSE for each element in list_a against each element in list_b
mse_matrix = np.mean((list_a[:, np.newaxis] - list_a[np.newaxis, :]) ** 2, axis=2)
off_diagonal_mse = mse_matrix[~np.eye(mse_matrix.shape[0], dtype=bool)]
upper_triangular_mse = mse_matrix[np.triu_indices(mse_matrix.shape[0], k=1)]
# print(off_diagonal_mse)
print(mse_matrix.flatten())
print(off_diagonal_mse.flatten())
print(upper_triangular_mse)
