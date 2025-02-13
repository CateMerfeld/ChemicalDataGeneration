import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import wandb
from sklearn.decomposition import PCA
import itertools
import GPUtil

import dask.dataframe as dd
import pandas as pd
import torch
import os
import random
import psutil
import time
import threading

from scipy.stats import zscore
from scipy.spatial import ConvexHull


# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------

# def create_individual_chemical_dataset_tensors(carl_dataset, embedding_preds, device, chem, multiple_carls_per_spec=True):
def create_individual_chemical_dataset_tensors(carl_dataset_file_path, embedding_preds_file_path, device=None, chem=None, multiple_carls_per_spec=True):
    carl_dataset = pd.read_feather(carl_dataset_file_path)
    carl_dataset.drop('level_0', axis=1, inplace=True)
    chem_carl_dataset = carl_dataset[carl_dataset['Label'] == chem]
    # del carl_dataset
    embedding_preds = pd.read_csv(embedding_preds_file_path)
    chem_embedding_preds = embedding_preds[embedding_preds[chem] == 1.0]
    # del embedding_preds
    return create_dataset_tensors_for_generator(chem_carl_dataset, chem_embedding_preds, device, multiple_carls_per_spec)

def flatten_and_bin(predicted_embeddings_batches):
    """
    Flatten prediction batches and convert to binary format.

    This function takes a list of batches containing predicted embeddings,
    flattens them, and converts each embedding into a binary vector where 
    the index of the maximum value is set to 1, and all other indices are 0.

    Parameters:
    ----------
    predicted_embeddings_batches : list of list of torch.Tensor
        Batches of predicted embeddings, with each embedding represented as a tensor.

    Returns:
    -------
    list of list of int
        A list of binary vectors corresponding to each embedding, with a 1 at 
        the index of the maximum value.
    """
    binary_preds_list = []
    
    for batch in predicted_embeddings_batches:
        for encoding in batch:
            # Get the index of the maximum value
            max_index = torch.argmax(encoding)
            # Create a binary label with 1 in the index of the highest value's index in the encoding 0s in all other indices
            binary_pred = [0] * len(encoding)
            binary_pred[max_index] = 1
            binary_preds_list.append(binary_pred)
    
    return binary_preds_list

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------

def run_with_wandb(config, **kwargs):
    """
    Initialize a WandB run with the given configuration.

    This function updates the provided configuration with additional keyword 
    arguments, initializes a WandB run, sets the number of threads for PyTorch, 
    and determines the device (GPU or CPU) to be used for computations.

    Parameters:
    ----------
    config : dict
        Configuration dictionary containing WandB settings and other parameters.
        Must include 'wandb_entity', 'wandb_project', and 'threads'.

    **kwargs : keyword arguments
        Additional configuration parameters to be added to the `config`.
    """
    config.update(kwargs)

    wandb.init(entity=config['wandb_entity'],
               project=config['wandb_project'],
               config=config)

    # Set the number of threads
    torch.set_num_threads(config['threads'])

    # Find out is there is a GPU available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not config['gpu']:
        device = torch.device('cpu')
    print(f'Using device: {device}')

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
 

def update_wandb_kwargs(wandb_kwargs, updates):
    """
    Update a dictionary of WandB keyword arguments with new values.

    Parameters:
    ----------
    wandb_kwargs : dict
        The original dictionary of WandB keyword arguments to be updated.

    updates : dict
        A dictionary containing new values to update in `wandb_kwargs`.

    Returns:
    -------
    dict
        The updated dictionary of WandB keyword arguments.
    """
    for key in updates.keys():
        wandb_kwargs[key] = updates[key]
    return wandb_kwargs

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------

def train_one_epoch(
        train_dataset, device, model, criterion, optimizer, 
        epoch, combo
        ):
  """
    Train the model for one epoch on the given training dataset.

    This function performs forward and backward passes for each batch in the 
    training dataset, computes the loss, and updates the model weights. 
    It also collects predicted embeddings and name encodings at the last epoch.

    Parameters:
    ----------
    train_dataset : iterable
        An iterable dataset that yields batches of data, name encodings, true 
        embeddings, and additional information.

    device : torch.device
        The device (CPU or GPU) on which to perform the training.

    model : torch.nn.Module
        The model to be trained.

    criterion : callable
        The loss function used to compute the loss.

    optimizer : torch.optim.Optimizer
        The optimizer used to update the model weights.

    epoch : int
        The current epoch number.

    combo : dict
        A dictionary containing configuration settings, including 'epochs'.

    Returns:
    -------
    float
        The average training loss for the epoch. If it's the last epoch, 
        also returns the predicted embeddings and name encodings.

    tuple (float, list, list)
        At last epoch (either final or early stopping), returns a tuple containing the average loss, 
        a list of predicted embeddings, and a list of corresponding name encodings.

  """
  epoch_training_loss = 0

  predicted_embeddings = []
  output_name_encodings = []

  for batch, name_encodings, true_embeddings, _ in train_dataset:
    # move inputs to device
    batch = batch.to(device)
    name_encodings = name_encodings.to(device)
    true_embeddings = true_embeddings.to(device)

    # backprapogation
    optimizer.zero_grad()
    
    batch_predicted_embeddings = model(batch)

    loss = criterion(batch_predicted_embeddings, true_embeddings)
    # accumulate epoch training loss
    epoch_training_loss += loss.item()

    loss.backward()
    optimizer.step()

    # at last epoch store output embeddings and corresponding labels to output list
    if (epoch + 1) == combo['epochs']:
      output_name_encodings.append(name_encodings)
      predicted_embeddings.append(batch_predicted_embeddings)

  # divide by number of batches to calculate average loss
  average_loss = epoch_training_loss/len(train_dataset)
  if (epoch + 1) == combo['epochs']:
    return average_loss, predicted_embeddings, output_name_encodings
  else:
    return average_loss
  
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
 
def preds_to_emb_pca_plot(
        predicted_embeddings, output_name_encodings, 
        sorted_chem_names, emb_df, 
        mass_spec_encoder_embeddings=False, mass_spec_chems=False
        ):
    """
    Generate and return data for PCA visualization of predicted embeddings alongside ChemNet embeddings.

    This function flattens the predicted embeddings and their corresponding chemical names, 
    optionally includes mass spectrometry embeddings, and prepares data for PCA plotting.

    Parameters:
    ----------
    predicted_embeddings : list of list of torch.Tensor
        A nested list of predicted embeddings, where each inner list contains tensors for a batch.

    output_name_encodings : list of list of torch.Tensor
        A nested list of one-hot encoded tensors representing the chemical names for the predicted embeddings.

    sorted_chem_names : list of str
        A list of chemical names corresponding to the indices of the one-hot encodings.

    emb_df : pandas.DataFrame
        A DataFrame containing true embeddings, with 'Embedding Floats' as one of its columns.

    mass_spec_encoder_embeddings : bool, optional
        If True, includes mass spectrometry encoder embeddings in the output.

    mass_spec_chems : list of str, optional
        A list of chemical names corresponding to mass spectrometry embeddings.

    Returns:
    -------
    tuple
        A tuple containing:
        - true_embeddings (pd.DataFrame): DataFrame of true embeddings used for comparison.
        - predicted_embeddings_flattened (list): Flattened list of predicted embeddings.
        - chem_names (list): List of chemical names corresponding to the predicted embeddings.
    """
    try:
        # Currently, preds and name encodings are lists of [n_batches, batch_size], flattening to lists of [n_samples]
        predicted_embeddings_flattened = [emb.cpu().detach().numpy() for emb_list in predicted_embeddings for emb in emb_list]
        chem_name_encodings_flattened = [enc.cpu() for enc_list in output_name_encodings for enc in enc_list]
    except AttributeError as e:
        predicted_embeddings_flattened = [emb for emb_list in predicted_embeddings for emb in emb_list]
        chem_name_encodings_flattened = [enc for enc_list in output_name_encodings for enc in enc_list]

    # Get chemical names from encodings
    chem_names = [sorted_chem_names[list(encoding).index(1)] for encoding in chem_name_encodings_flattened]

    if mass_spec_encoder_embeddings:
        for emb in mass_spec_encoder_embeddings:
            predicted_embeddings_flattened.append(torch.Tensor(emb))
        chem_names += mass_spec_chems

    try:
        # making list of all embeddings and chem names except for BKG
        embeddings = [emb for emb in emb_df['Embedding Floats']][1:]
        cols = emb_df.index[1:]
        true_embeddings = pd.DataFrame(embeddings).T
        true_embeddings.columns = cols
        
    except KeyError as e:
        if str(e) == "'Embedding Floats'":
            true_embeddings = emb_df
    
    return (true_embeddings, predicted_embeddings_flattened, chem_names)

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
 
def predict_embeddings(dataset, model, device, criterion, reparameterization=False):
    """
    Generate predicted embeddings and compute average loss on the given dataset.

    This function evaluates the model on the provided dataset, computes the predicted 
    embeddings, and calculates the average loss by comparing predictions to true embeddings.

    Parameters:
    ----------
    dataset : iterable
        An iterable dataset that yields batches of data, name encodings, true embeddings, 
        and spectra indices.

    model : torch.nn.Module
        The model to be evaluated.

    device : torch.device
        The device (CPU or GPU) on which to perform the evaluations.

    criterion : callable
        The loss function used to compute the loss between predicted and true embeddings.

    Returns:
    -------
    tuple
        A tuple containing:
        - predicted_embeddings (list): List of predicted embeddings for each batch.
        - output_name_encodings (list): List of name encodings for the predicted embeddings.
        - average_loss (float): The average loss over all batches.
        - input_spectra_indices (list): List of spectra indices corresponding to the input data.
    """
    total_loss = 0

    model.eval() # Set model to evaluation mode
    predicted_embeddings = []
    output_name_encodings = []
    input_spectra_indices = []

    with torch.no_grad():
        for batch, name_encodings, true_embeddings, spectra_indices in dataset:
            batch = batch.to(device)
            true_embeddings = true_embeddings.to(device)

            batch_predicted_embeddings = model(batch, reparameterization)
            predicted_embeddings.append(batch_predicted_embeddings.to('cpu').detach().numpy())
            output_name_encodings.append(name_encodings.to('cpu').detach().numpy())
            input_spectra_indices.append(spectra_indices.to('cpu').detach().numpy())

            # print(batch_predicted_embeddings.shape, true_embeddings.shape)

            loss = criterion(batch_predicted_embeddings, true_embeddings)
            # accumulate loss
            total_loss += loss.item()

    # divide by number of batches to calculate average loss
    average_loss = total_loss/len(dataset)
    return predicted_embeddings, output_name_encodings, average_loss, input_spectra_indices

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
def add_hulls(ax, pca, chem_data, threshold=3):
    # embeddings_by_bkg_list = [chem_data.iloc[::2], chem_data.iloc[1::2]]
    # for df in embeddings_by_bkg_list:
    z_scores = np.abs(zscore(chem_data))
    threshold = threshold  # Adjust threshold as needed

    # Exclude outliers
    filtered_data = chem_data[(z_scores < threshold).all(axis=1)]
    # print(f'Excluded {round((len(chem_data) - len(filtered_data))/len(chem_data)*100)}% of data points as outliers.')

    transformed_data = pca.transform(filtered_data)
    hull = ConvexHull(transformed_data)
    for simplex in hull.simplices:
        ax.plot(transformed_data[simplex, 0], transformed_data[simplex, 1], 'r-') 
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
 
def plot_emb_pca(
        all_embeddings, ims_embeddings, results_type, input_type, embedding_type='ChemNet', mass_spec_embeddings = None, log_wandb=True, 
        chemnet_embeddings_to_plot=None, mse_insert=None, insert_position=[0.05, 0.05], show_wandb_run_name=True, plot_hulls=False, hull_data=None):
    """
    This function performs Principal Component Analysis (PCA) on chemical embeddings and visualizes the results
    in a 2D scatter plot. It overlays additional data from ion mobility spectrometry (IMS) and mass spectrometry 
    if provided. The plot includes legends for different markers and can display the mean squared error (MSE) 
    and Weights & Biases (WandB) run name.

    Parameters:
    ----------
    all_embeddings : pd.DataFrame
        DataFrame containing all chemical embeddings to be plotted.

    ims_embeddings : pd.DataFrame
        DataFrame containing IMS embeddings, including a 'Label' column for chemical names.

    results_type : str
        A string indicating the type of results being plotted (train, val or test), used for title annotation.

    input_type : str
        A string indicating the type of input data (IMS, Carl or MNIST), used for legend annotation.

    embedding_type : str, optional
        A string specifying the type of embedding being plotted (ChemNet or OneHot). Default is 'ChemNet'.

    mass_spec_embeddings : pd.DataFrame, optional
        DataFrame containing mass spectrometry embeddings, including a 'Label' column. Default is None.

    log_wandb : bool, optional
        If True, logs the plot to Weights & Biases (WandB). Default is False.

    chemnet_embeddings_to_plot : pd.DataFrame, optional
        DataFrame containing specific ChemNet embeddings to plot. Default is None, which means all embeddings will be used.

    mse_insert : float, optional
        The mean squared error value to display in the plot. Default is None.

    insert_position : list of float, optional
        A list specifying the position to insert the MSE text in the plot, given as [x, y] in axis coordinates. Default is [0.05, 0.05].

    show_wandb_run_name : bool, optional
        If True, includes the current WandB run name in the plot. Default is True.

    Returns:
    -------
    None
        The function displays a PCA plot and optionally logs it to WandB.
    """
    pca = PCA(n_components=2)
    pca.fit(all_embeddings.T)

    if chemnet_embeddings_to_plot is not None:
        transformed_embeddings = pca.transform(chemnet_embeddings_to_plot.T)
        all_chemical_names = list(chemnet_embeddings_to_plot.columns)
    else:
        transformed_embeddings = pca.transform(all_embeddings.T) 
        all_chemical_names = list(all_embeddings.columns)

    _, ax = plt.subplots(figsize=(8,6))

    # Create a color cycle for distinct colors
    color_cycle = plt.gca()._get_lines.prop_cycler

    ims_labels = list(ims_embeddings['Label'])
    if mass_spec_embeddings is not None:
        mass_spec_labels=list(mass_spec_embeddings['Label'])
    else:
        mass_spec_labels = False
    
    # Scatter plot
    for chem in all_chemical_names:
        idx = all_chemical_names.index(chem)
        color = next(color_cycle)['color']
        # Plot ChemNet embeddings
        if idx < 8: # only label 1st 8 chemicals to avoid giant legend
            ax.scatter(transformed_embeddings[idx, 0], transformed_embeddings[idx, 1], color = color, label=chem)#, s=200)
        else:
            ax.scatter(transformed_embeddings[idx, 0], transformed_embeddings[idx, 1], color = color)#, s=75)

        # Transform encoder-generated ims_embeddings for the current chemical, if we have ims data for chem
        if chem in ims_labels:
            # transform all data for the given chemical. Exclude last col (label)
            ims_transformed = pca.transform(ims_embeddings[ims_embeddings['Label'] == chem].iloc[:, :-1])
            
            # Scatter plot for ims_embeddings with a different marker
            ax.scatter(ims_transformed[:, 0], ims_transformed[:, 1], marker='o', facecolors='none', edgecolors=color)#marker='x', color=color)#, s=75)
            if plot_hulls:
                if hull_data is None:
                    hull_data = ims_embeddings
                add_hulls(ax, pca, hull_data[hull_data['Label'] == chem].iloc[:, :-1])


        # repeat for mass spec
        if mass_spec_labels:
            if chem in mass_spec_labels:
                # transform all data for the given chemical. Exclude last col (label)
                mass_spec_transformed = pca.transform(mass_spec_embeddings[mass_spec_embeddings['Label'] == chem].iloc[:, :-1].values)
                
                # Scatter plot for mass_spec_embeddings with a different marker
                ax.scatter(mass_spec_transformed[:, 0], mass_spec_transformed[:, 1], marker='*', color=color, s=75)
    # Add legend
    legend1 = ax.legend(loc='upper right', title='Label')
    ax.add_artist(legend1)

    marker_legends = [
    plt.Line2D([0], [0], marker='o', color='w', label=embedding_type, markerfacecolor='black', markersize=6),
    plt.Line2D([0], [0], marker='o', color='w', label=input_type, markerfacecolor='none', markeredgecolor='black', markersize=6),
    ]
    
    if mass_spec_embeddings is not None:
        marker_legends.append(plt.Line2D([0], [0], marker='*', color='w', label='Mass Spec', markerfacecolor='black', markersize=10))

    # Add the second legend
    legend2 = ax.legend(handles=marker_legends, title='Marker Types', loc='upper left')
    ax.add_artist(legend2)

    if mse_insert is not None:
        # Add mse text in the corner with a box
        plt.text(insert_position[0], insert_position[1], f'MSE: {format(mse_insert, ".2e")}', 
            transform=plt.gca().transAxes,  # Use axis coordinates
            fontsize=14,
            verticalalignment='bottom',  # Align text to the top
            horizontalalignment='left',  # Align text to the right
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))  # Box properties
    
    if show_wandb_run_name == True:
        run_name = wandb.run.name
        # Add wandb run text in the corner
        xlim = plt.xlim()
        ylim = plt.ylim()
        plt.text(xlim[1] - 0.01 * (xlim[1] - xlim[0]),  # x position with an offset
                ylim[0] + 0.01 * (ylim[1] - ylim[0]),  # y position with an offset
                f'WandB run: {run_name}', 
                fontsize=8,
                verticalalignment='bottom',  # Align text to the top
                horizontalalignment='right',  # Align text to the right
                bbox=dict(facecolor='white', alpha=0.001, edgecolor='white'))

    plt.xticks([])
    plt.yticks([])
    if embedding_type != 'ChemNet':
        plt.title(f'{embedding_type} vs. Encoder {results_type} Output PCA', fontsize=18)
    else:
        plt.title(f'ChemNet vs. Encoder {results_type} Output PCA', fontsize=18)

    if log_wandb:
        plt.savefig('tmp_plot.png', format='png', dpi=300)
        wandb.log({'PCA of Predicted Chemical Embeddings': wandb.Image('tmp_plot.png')})

    plt.show()

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------

def plot_generation_results_pca(
        true_spectra, synthetic_spectra, chem_labels, results_type, sample_size=None, 
        chem_of_interest=None, log_wandb=False, mse_insert=None, 
        insert_position=[0.05, 0.05], show_wandb_run_name=False):
    
    pca = PCA(n_components=2)
    # assert true_spectra.shape[1] == 1677, "True spectra df does not have 1677 columns"
    pca.fit(true_spectra.iloc[:,:-1])

    _, ax = plt.subplots(figsize=(8,6))

    # Create a color cycle for distinct colors
    color_cycle = plt.gca()._get_lines.prop_cycler

    if sample_size is not None:
        true_spectra = true_spectra.sample(n = sample_size, random_state=42)

    # Scatter plot
    for chem, color in zip(chem_labels, color_cycle):
        # color = next(color_cycle)['color']
        color = color['color']
        # Transform data for the current chemical, exclude last col (label)
        transformed_true_spectra = pca.transform(true_spectra[true_spectra['Label'] == chem].iloc[:, :-1])
        # if specified, make makers larger for the chemical of interest
        if chem_of_interest is not None:
            if chem != chem_of_interest:
                ax.scatter(transformed_true_spectra[:, 0], transformed_true_spectra[:, 1], marker='o', label=chem, color=color, s=10)
            else:
                ax.scatter(transformed_true_spectra[:, 0], transformed_true_spectra[:, 1], marker='o', label=chem, color=color, s=100)
        else:
            ax.scatter(transformed_true_spectra[:, 0], transformed_true_spectra[:, 1], marker='o', label=chem, color=color)

        synthetic_chem = synthetic_spectra[synthetic_spectra['Label'] == chem].iloc[:, :-1]
        # only plot synthetic spectra if there are any for given chemical
        if synthetic_chem.shape[0] > 0:
            transformed_synthetic_spectra = pca.transform(synthetic_chem)
            # Scatter plot for synthetic spectra with a different marker
            ax.scatter(transformed_synthetic_spectra[:, 0], transformed_synthetic_spectra[:, 1], marker='*', color=color)

    # Add legend
    legend1 = ax.legend(loc='upper right', title='Label')
    ax.add_artist(legend1)

    marker_legends = [
    plt.Line2D([0], [0], marker='o', color='w', label='Experimental Spectra', markerfacecolor='black', markersize=6),
    plt.Line2D([0], [0], marker='*', color='w', label='Synthetic Spectra', markerfacecolor='black', markersize=10),
    ]
    

    # Add the second legend
    legend2 = ax.legend(handles=marker_legends, title='Marker Types', loc='upper left')
    ax.add_artist(legend2)

    if mse_insert is not None:
        # Add mse text in the corner with a box
        plt.text(insert_position[0], insert_position[1], f'MSE: {format(mse_insert, ".2e")}', 
            transform=plt.gca().transAxes,  # Use axis coordinates
            fontsize=14,
            verticalalignment='bottom',  # Align text to the top
            horizontalalignment='left',  # Align text to the right
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))  # Box properties
    
    if show_wandb_run_name == True:
        run_name = wandb.run.name
        # Add wandb run text in the corner
        xlim = plt.xlim()
        ylim = plt.ylim()
        plt.text(xlim[1] - 0.01 * (xlim[1] - xlim[0]),  # x position with an offset
                ylim[0] + 0.01 * (ylim[1] - ylim[0]),  # y position with an offset
                f'WandB run: {run_name}', 
                fontsize=8,
                verticalalignment='bottom',  # Align text to the top
                horizontalalignment='right',  # Align text to the right
                bbox=dict(facecolor='white', alpha=0.001, edgecolor='white'))

    plt.xticks([])
    plt.yticks([])
    plt.title(f'Experimental vs. Synthetic {results_type} Spectra PCA', fontsize=18)

    if log_wandb:
        plt.savefig('tmp_plot.png', format='png', dpi=300)
        wandb.log({'PCA of Experimental vs. Synthetic Spectra': wandb.Image('tmp_plot.png')})

    plt.show()


# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------

def plot_generation_results_pca_single_chem_side_by_side(
        true_spectra, synthetic_spectra, chem_labels, results_type, sample_size=None, 
        chem_of_interest=None, log_wandb=False, mse_insert=None, 
        insert_position=[0.05, 0.05], show_wandb_run_name=False, 
        x_lims=None, y_lims=None, save_plot_path=None):
    
    # if pca is None:
    pca = PCA(n_components=2)
    pca.fit(true_spectra.iloc[:,:-1])

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Setting x and y limits so that plot scales are the same
    if x_lims is not None:
        ax1.set_xlim(x_lims[0], x_lims[1])
        ax2.set_xlim(x_lims[0], x_lims[1])
    if y_lims is not None:
        ax1.set_ylim(y_lims[0], y_lims[1])
        ax2.set_ylim(y_lims[0], y_lims[1])
    if x_lims is None: # if plot boundaries are not specified, set them so that scales are consistent between plots
        true_transformed = pca.transform(true_spectra.iloc[:, :-1])
        synthetic_transformed = pca.transform(synthetic_spectra.iloc[:, :-1])
        x_lims = [
            min(true_transformed[:, 0].min(), synthetic_transformed[:, 0].min()) * 1.2, 
            max(true_transformed[:, 0].max(), synthetic_transformed[:, 0].max()) * 1.2
        ]
        y_lims = [
            min(true_transformed[:, 1].min(), synthetic_transformed[:, 1].min()) * 1.2,
            max(true_transformed[:, 1].max(), synthetic_transformed[:, 1].max()) * 1.2
        ]
    ax1.set_xlim(x_lims[0], x_lims[1])
    ax2.set_xlim(x_lims[0], x_lims[1])
    ax1.set_ylim(y_lims[0], y_lims[1])
    ax2.set_ylim(y_lims[0], y_lims[1])

    # Create a color cycle for distinct colors
    color_cycle = plt.gca()._get_lines.prop_cycler

    if sample_size is not None:
        true_spectra = true_spectra.sample(n=sample_size, random_state=42)

    # Plot for true spectra
    for chem, color in zip(chem_labels, color_cycle):
        color = color['color']
        transformed_true_spectra = pca.transform(true_spectra[true_spectra['Label'] == chem].iloc[:, :-1])
        
        if chem_of_interest is not None:
            marker_size = 50 if chem == chem_of_interest else 10
            if chem == chem_of_interest:
                ax1.scatter(transformed_true_spectra[:, 0], transformed_true_spectra[:, 1], marker='o', label=chem, color=color, s=marker_size)
            else:
                true_sample = true_spectra[true_spectra['Label'] == chem].iloc[:, :-1].sample(n=10, random_state=42)
                transformed_sample = pca.transform(true_sample)
                ax1.scatter(transformed_sample[:, 0], transformed_sample[:, 1], marker='o', label=chem, color=color, s=marker_size)
                ax2.scatter(transformed_sample[:, 0], transformed_sample[:, 1], marker='o', label=chem, color=color, s=marker_size)
        # ax2.scatter(transformed_true_spectra[:, 0], transformed_true_spectra[:, 1], marker='o', color=color, s=1)
            # ax1.scatter(transformed_true_spectra[:, 0], transformed_true_spectra[:, 1], marker='o', label=chem, color=color)
        # else:
        #     ax1.scatter(transformed_true_spectra[:, 0], transformed_true_spectra[:, 1], marker='o', label=chem, color=color)
        
        synthetic_chem = synthetic_spectra[synthetic_spectra['Label'] == chem].iloc[:, :-1]
        
        if synthetic_chem.shape[0] > 0:
            transformed_synthetic_spectra = pca.transform(synthetic_chem)
            ax2.scatter(transformed_synthetic_spectra[:, 0], transformed_synthetic_spectra[:, 1], marker='o', label=chem, color=color, s=50)
    
    # if chem_of_interest is not None:
    ax1.set_title(f'Experimental {results_type} Spectra PCA {chem_of_interest}', fontsize=18)
    ax2.set_title(f'Synthetic {results_type} Spectra PCA {chem_of_interest}', fontsize=18)
    # else:
    #     ax1.set_title(f'Experimental {results_type} Spectra PCA', fontsize=18)
    #     ax2.set_title(f'Synthetic {results_type} Spectra PCA', fontsize=18)
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2.set_xticks([])
    ax2.set_yticks([])

    # Add legends
    ax1.legend(loc='upper right', title='Label')
    ax2.legend(loc='upper right', title='Label')

    if mse_insert is not None:
        plt.text(insert_position[0], insert_position[1], f'MSE: {format(mse_insert, ".2e")}', 
                 transform=plt.gca().transAxes, fontsize=14, verticalalignment='bottom',
                 horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))  
    
    if show_wandb_run_name:
        run_name = wandb.run.name
        xlim = plt.xlim()
        ylim = plt.ylim()
        plt.text(xlim[1] - 0.01 * (xlim[1] - xlim[0]), ylim[0] + 0.01 * (ylim[1] - ylim[0]),
                 f'WandB run: {run_name}', fontsize=8, verticalalignment='bottom',
                 horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.001, edgecolor='white'))

    plt.tight_layout()

    if log_wandb:
        plt.savefig('tmp_plot.png', format='png', dpi=300)
        wandb.log({'PCA of Experimental vs. Synthetic Spectra': wandb.Image('tmp_plot.png')})
    if save_plot_path is not None:
        plot_path = os.path.join(save_plot_path, f'{chem_of_interest}_real_synthetic_pca_comparison.png')
        plt.savefig(plot_path, format='png', dpi=300)

    plt.show()

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
 
def plot_pca(
    data, batch_size, model, device, encoder_criterion, sorted_chem_names, 
    all_embeddings_df, ims_embeddings_df, results_type, 
    input_type, embedding_type='ChemNet',
    show_wandb_run_name=True, log_wandb=True, 
    ):
    """
    Perform PCA on chemical embeddings and plot the transformed data.

    This function generates a PCA scatter plot for ChemNet embeddings, 
    including IMS and mass spectrometry embeddings if provided.

    Parameters:
    ----------
    all_embeddings : pd.DataFrame
        DataFrame containing ChemNet embeddings for all chemicals, with each column 
        representing one chemical's embedding.

    ims_embeddings : pd.DataFrame
        DataFrame containing IMS (ion mobility spectrometry) embeddings, must include 
        a 'Label' column with chemical names.

    mass_spec_embeddings : pd.DataFrame, optional
        DataFrame containing mass spectrometry embeddings, similar structure to `ims_embeddings`.
        Default is None.

    log_wandb : bool, optional
        If True, logs the generated plot to Weights & Biases (wandb). Default is False.

    chemnet_embeddings_to_plot : pd.DataFrame, optional
        DataFrame containing ChemNet embeddings specifically to be plotted.
    
    results_type: str
        Type of results - train, val, or test

    input_type : str
        The type of input data - IMS, Carl, MNIST, etc

    embedding_type : str, optional
        The type of embedding being visualized - ChemNet, OneHot, etc. Default is ChemNet.

    mse_insert : float, optional
        Mean Squared Error value to display on the plot.

    insert_position : list of float, optional
        Location in axis coordinates for MSE text insertion. Default is [0.05, 0.05].

    show_wandb_run_name : bool, optional
        If True, displays the current WandB run name on the plot. Default is True.

    Returns:
    -------
    None
        Displays the PCA scatter plot with ChemNet, IMS, and mass spec embeddings.

    Notes:
    -----
    - PCA is performed on the transpose of `all_embeddings` to align with IMS and mass spec data.
    """
    dataset = DataLoader(
        data, 
        batch_size=batch_size, 
        shuffle=False
    )

    preds, name_encodings, avg_loss, _ = predict_embeddings(dataset, model, device, encoder_criterion)
    true_embeddings, predicted_embeddings_flattened, chem_names = preds_to_emb_pca_plot(
        preds, name_encodings, sorted_chem_names, ims_embeddings_df,  
        )
    preds_df = pd.DataFrame(predicted_embeddings_flattened)
    preds_df['Label'] = chem_names
    
    plot_emb_pca(
        all_embeddings_df, preds_df, results_type=results_type, input_type=input_type,
        embedding_type=embedding_type, log_wandb=log_wandb, 
        chemnet_embeddings_to_plot=true_embeddings, mse_insert=avg_loss,
        show_wandb_run_name=show_wandb_run_name
        )

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------

def plot_carl_real_synthetic_comparison(
        true_carl, synthetic_carl, results_type, chem_label, 
        log_wandb=False, show_wandb_run_name=False, criterion=None, 
        run_name=None, save_plot_path=None):
    
    _, axes = plt.subplots(1, 2, figsize=(14, 8))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # x axis should run from lowest drift time (184) to highest drift time (184 + len(true_carl)//2)
    numbers = range(184, (len(true_carl)//2)+184)
    # y axis should run from min of both carls to max of both carls
    min_y = min(min(list(true_carl)), min(list(synthetic_carl))) - 200
    max_y = max(max(list(true_carl)), max(list(synthetic_carl))) + 200

    axes[0].plot(numbers, true_carl[:len(numbers)], label='Positive')
    axes[0].plot(numbers, true_carl[len(numbers):], label='Negative')
    axes[0].set_title(f'True {results_type} {chem_label} CARL', fontsize=20)
    axes[0].set_xlabel('Drift Time', fontsize=16)
    axes[0].set_ylabel('Ion Intensity', fontsize=16)
    axes[0].set_ylim(min_y, max_y)
    axes[0].legend(fontsize=14)

    axes[1].plot(numbers, synthetic_carl[:len(numbers)], label='Positive')
    axes[1].plot(numbers, synthetic_carl[len(numbers):], label='Negative')
    axes[1].set_title(f'Synthetic {results_type} {chem_label} CARL', fontsize=20)
    axes[1].set_xlabel('Drift Time', fontsize=16)
    axes[1].set_ylabel('Ion Intensity', fontsize=16)
    axes[1].set_ylim(min_y, max_y)
    axes[1].legend(fontsize=14)

    if criterion is not None:
        xlim = axes[0].get_xlim()
        ylim = axes[0].get_ylim()
        axes[0].text(xlim[1] - 0.02 * (xlim[1] - xlim[0]),  # x position with an offset
                    ylim[0] + 0.05 * (ylim[1] - ylim[0]),  # y position with an offset 
                    'Real vs. Synthetic MSE: {:.2e}'.format(criterion(torch.Tensor(true_carl), torch.Tensor(synthetic_carl)).item()),
                    fontsize=14,
                    verticalalignment='bottom',  # Align text to the bottom
                    horizontalalignment='right',  # Align text to the left
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))

    if show_wandb_run_name == True:
        if run_name is None:
            run_name = wandb.run.name
        # Add wandb run text in the corner
        xlim = plt.xlim()
        ylim = plt.ylim()
        plt.text(xlim[1] - 0.01 * (xlim[1] - xlim[0]),  # x position with an offset
                ylim[0] + 0.01 * (ylim[1] - ylim[0]),  # y position with an offset
                f'WandB run: {run_name}', 
                fontsize=8,
                verticalalignment='bottom',  # Align text to the bottom
                horizontalalignment='right',  # Align text to the right
                bbox=dict(facecolor='white', alpha=0.001, edgecolor='white'))

    if log_wandb:
        plt.savefig('tmp_plot.png', format='png', dpi=300)
        wandb.log({'Comparison of Real and Synthetic CARLs': wandb.Image('tmp_plot.png')})

    plt.tight_layout()

    if save_plot_path is not None:
        plt.savefig(save_plot_path, format='png', dpi=300)

    plt.show()

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------

def format_data_for_plotting(data):
    assert data.columns[0] == 'Unnamed: 0', 'First column should be "Unnamed: 0"'
    if data.shape[1] == 1679:
        assert data.columns[-1] == 'Label', 'Last column should be "Label"'
        labels = list(data['Label'])
        return data.iloc[:,2:], labels
    if data.shape[1] == 1687:
        assert data.columns[-1] == 'TEPO', 'Last column should be "TEPO"'
        sorted_chem_names = data.columns[-8:]
        labels = list(data['Label'])
        return data.iloc[:,2:-8], sorted_chem_names, labels

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------

def plot_spectra_real_synthetic_comparison(true_spec, synthetic_spec, results_type, chem_label, log_wandb=False, show_wandb_run_name=True, criterion=None, run_name=None):
    _, axes = plt.subplots(1, 2, figsize=(14, 8))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # x axis should run from lowest drift time (184) to highest drift time (184 + len(true_carl)//2)
    numbers = range(184, (len(true_spec)//2)+184)
    # y axis should run from min of both carls to max of both carls
    min_y = min(list(true_spec)+list(synthetic_spec)) - 200
    max_y = max(list(true_spec)+list(synthetic_spec)) + 200

    axes[0].plot(numbers, true_spec[:len(numbers)], label='Positive')
    axes[0].plot(numbers, true_spec[len(numbers):], label='Negative')
    axes[0].set_title(f'Experimental {chem_label} {results_type} Spectrum', fontsize=20)
    axes[0].set_xlabel('Drift Time', fontsize=16)
    axes[0].set_ylabel('Ion Intensity', fontsize=16)
    axes[0].set_ylim(min_y, max_y)
    axes[0].legend(fontsize=14)

    axes[1].plot(numbers, synthetic_spec[:len(numbers)], label='Positive')
    axes[1].plot(numbers, synthetic_spec[len(numbers):], label='Negative')
    axes[1].set_title(f'Synthetic {chem_label} {results_type} Spectrum', fontsize=20)
    axes[1].set_xlabel('Drift Time', fontsize=16)
    axes[1].set_ylabel('Ion Intensity', fontsize=16)
    axes[1].set_ylim(min_y, max_y)
    axes[1].legend(fontsize=14)

    if criterion is not None:
        xlim = axes[0].get_xlim()
        ylim = axes[0].get_ylim()
        axes[0].text(xlim[1] - 0.02 * (xlim[1] - xlim[0]),  # x position with an offset
                    ylim[0] + 0.05 * (ylim[1] - ylim[0]),  # y position with an offset 
                    'Real vs. Synthetic MSE: {:.2e}'.format(criterion(torch.Tensor(true_spec), torch.Tensor(synthetic_spec)).item()),
                    fontsize=14,
                    verticalalignment='bottom',  # Align text to the bottom
                    horizontalalignment='right',  # Align text to the left
                    bbox=dict(facecolor='white', alpha=1, edgecolor='black'))

    if show_wandb_run_name == True:
        if run_name is None:
            run_name = wandb.run.name
        # Add wandb run text in the corner
        xlim = plt.xlim()
        ylim = plt.ylim()
        plt.text(xlim[1] - 0.01 * (xlim[1] - xlim[0]),  # x position with an offset
                ylim[0] + 0.01 * (ylim[1] - ylim[0]),  # y position with an offset
                f'WandB run: {run_name}', 
                fontsize=8,
                verticalalignment='bottom',  # Align text to the bottom
                horizontalalignment='right',  # Align text to the right
                bbox=dict(facecolor='white', alpha=0.001, edgecolor='white'))

    if log_wandb:
        plt.savefig('tmp_plot.png', format='png', dpi=300)
        wandb.log({'Comparison of Experimental and Synthetic CARLs': wandb.Image('tmp_plot.png')})

    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------

class Encoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.encoder = nn.Sequential(
      nn.Linear(1676,1548),
      nn.LeakyReLU(inplace=True),
      nn.Linear(1548,1420),
      nn.LeakyReLU(inplace=True),
      nn.Linear(1420, 1292),
      nn.LeakyReLU(inplace=True),
      nn.Linear(1292, 1164),
      nn.LeakyReLU(inplace=True),
      nn.Linear(1164, 1036),
      nn.LeakyReLU(inplace=True),
      nn.Linear(1036, 908),
      nn.LeakyReLU(inplace=True),
      nn.Linear(908, 780),
      nn.LeakyReLU(inplace=True),
      nn.Linear(780, 652),
      nn.LeakyReLU(inplace=True),
    )

    # self.final_relu = 
    self.final_linear = nn.Linear(652, 512)

    self.mean_layer = nn.Linear(652, 512)
    self.logvar_layer = nn.Linear(652, 512)

  def reparameterize(self, mean, log_var):
    eps = torch.randn_like(log_var)
    z = mean + log_var * eps
    return z

  def forward(self, x, reparameterization=False):
    x = self.encoder(x)
    
    # Do reparameterization if desired, otherwise run final encoder layer
    if reparameterization:
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        z = self.reparameterize(mean, logvar)
        return z
    else:
        x = self.final_linear(x)
        return x

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------

def train_model(
        model_type, train_data, val_data, test_data, device, config, wandb_kwargs, 
        all_embeddings_df, ims_embeddings_df, model_hyperparams, sorted_chem_names, 
        encoder_path, save_emb_pca_to_wandb = True, early_stop_threshold=10, 
        input_type='IMS', embedding_type='ChemNet', show_wandb_run_name=True, 
        lr_scheduler = False, patience=5
        ):
    
    """
    Train a model with specified hyperparameters and log results using Weights & Biases.

    Parameters:
    ----------
    model_type : str
        The type of model to be trained (e.g., 'Encoder').

    train_data : Dataset
        The dataset used for training the model.

    val_data : Dataset
        The dataset used for validating the model during training.

    test_data : Dataset
        The dataset used for evaluating the model after training.

    device : torch.device
        The device (CPU or GPU) on which to perform the training.

    config : dict
        Configuration settings for the training process.

    wandb_kwargs : dict
        Arguments for logging to Weights & Biases.

    all_embeddings_df : pd.DataFrame
        DataFrame containing all embeddings for chemicals.

    ims_embeddings_df : pd.DataFrame
        DataFrame containing IMS (ion mobility spectrometry) embeddings.

    model_hyperparams : dict
        Dictionary of hyperparameters for the model, with keys as parameter names and values as lists of options.

    sorted_chem_names : list of str
        List of sorted chemical names corresponding to the embeddings.

    encoder_path : str
        File path to save the best model state.

    save_emb_pca_to_wandb : bool, optional
        If True, saves PCA plots of embeddings to Weights & Biases. Default is True.

    early_stop_threshold : int, optional
        Number of epochs without improvement in validation loss before stopping training. Default is 10.

    input_type : str, optional
        The type of input being used (IMS, Carl, MNIST). Default is 'IMS'.

    embedding_type : str, optional
        The type of embedding being used (ChemNet, OneHot). Default is 'ChemNet'.

    show_wandb_run_name : bool, optional
        If True, displays the current WandB run name on the plot. Default is True.

    Returns:
    -------
    dict
        The best hyperparameters found during training.
    """

    # loss to compare for each model. Starting at infinity so it will be replaced by first model's first epoch loss 
    lowest_val_loss = np.inf

    keys = model_hyperparams.keys()
    values = model_hyperparams.values()

    # Generate all parameter combinations from model_config using itertools.product
    combinations = itertools.product(*values)

    # Iterate through each parameter combination and run model 
    for combo in combinations:
        # creating different var for model loss to use for early stopping
        lowest_val_model_loss = np.inf
        
        if model_type == 'Encoder':
            model = Encoder().to(device)

        if model_type == 'Generator':
            model = Generator().to(device)

        epochs_without_validation_improvement = 0
        combo = dict(zip(keys, combo))

        train_dataset = DataLoader(train_data, batch_size=combo['batch_size'], shuffle=True)
        val_dataset = DataLoader(val_data, batch_size=combo['batch_size'], shuffle=False)

        optimizer = torch.optim.AdamW(model.parameters(), lr = combo['learning_rate'])
        criterion = nn.MSELoss()

        final_lr = combo['learning_rate']

        if lr_scheduler:
            # Initialize the learning rate scheduler with patience of 5 epochs 
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=patience, factor=0.1, verbose=True)

        wandb_kwargs = update_wandb_kwargs(wandb_kwargs, combo)

        run_with_wandb(config, **wandb_kwargs)

        print('--------------------------')
        print('--------------------------')
        print('New run with hyperparameters:')
        for key in combo:
            print(key, ' : ', combo[key])

        for epoch in range(combo['epochs']):
            if epochs_without_validation_improvement < early_stop_threshold:
                model.train(True)

                # do a pass over the data
                # at last epoch get predicted embeddings and chem names
                if (epoch + 1) == combo['epochs']:
                    average_loss, _, _ = train_one_epoch(
                    train_dataset, device, model, criterion, 
                    optimizer, epoch, combo
                    )
                    # save output pca to weights and biases
                    if save_emb_pca_to_wandb:
                        # plot_pca gets predictions from trained model and plots them
                        plot_pca(
                            train_data, combo['batch_size'], model, device, 
                            criterion, sorted_chem_names, all_embeddings_df, 
                            ims_embeddings_df, 'Train', input_type, embedding_type, show_wandb_run_name
                            )
                        plot_pca(
                            test_data, combo['batch_size'], model, device, 
                            criterion, sorted_chem_names, all_embeddings_df,
                            ims_embeddings_df, 'Test', input_type, embedding_type, show_wandb_run_name
                            )
                else:
                    average_loss = train_one_epoch(
                    train_dataset, device, model, criterion, optimizer, epoch, combo
                    )

                epoch_val_loss = 0  
                # evaluate model on validation data
                model.eval() # Set model to evaluation mode
                with torch.no_grad():
                    for val_batch, val_name_encodings, val_true_embeddings, _ in val_dataset:
                        val_batch = val_batch.to(device)
                        val_name_encodings = val_name_encodings.to(device)
                        val_true_embeddings = val_true_embeddings.to(device)

                        val_batch_predicted_embeddings = model(val_batch)

                        val_loss = criterion(val_batch_predicted_embeddings, val_true_embeddings)
                        # accumulate epoch validation loss
                        epoch_val_loss += val_loss.item()

                # divide by number of batches to calculate average loss
                val_average_loss = epoch_val_loss/len(val_dataset)

                if lr_scheduler:
                    scheduler.step(val_average_loss)  # Pass the validation loss to the scheduler
                    # get the new learning rate (to give to wandb)
                    final_lr = optimizer.param_groups[0]['lr']

                if val_average_loss < lowest_val_model_loss:
                    # check if val loss is improving for this model
                    epochs_without_validation_improvement = 0
                    lowest_val_model_loss = val_average_loss

                    if val_average_loss < lowest_val_loss:
                        # if current epoch of current model is best performing (of all epochs and models so far), save model 
                        torch.save(model, encoder_path)
                        print(f'Saved best model at epoch {epoch+1}')
                        lowest_val_loss = val_average_loss
                        best_hyperparams = combo
                    else:
                        print(f'Model best validation loss at {epoch+1}')
                
                else:
                    epochs_without_validation_improvement += 1

                # log losses to wandb
                if model_type == 'Encoder':
                    wandb.log({"Encoder Training Loss": average_loss, "Encoder Validation Loss": val_average_loss})
                elif model_type == 'Generator':
                    wandb.log({"Generator Training Loss": average_loss, "Generator Validation Loss": val_average_loss})

                if (epoch + 1) % 10 == 0 or epoch == 0:
                    print('Epoch[{}/{}]:'.format(epoch+1, combo['epochs']))
                    print(f'   Training loss: {average_loss}')
                    print(f'   Validation loss: {val_average_loss}')
                    print('-------------------------------------------')
            else:
                print(f'Validation loss has not improved in {epochs_without_validation_improvement} epochs. Stopping training at epoch {epoch+1}.')
                wandb.log({'Early Stopping Ecoch':epoch+1})
                wandb.log({'Learning Rate at Final Epoch':final_lr})
                plot_pca(
                    train_data, combo['batch_size'], model, device, 
                    criterion, sorted_chem_names, all_embeddings_df, 
                    ims_embeddings_df, 'Train', input_type, embedding_type, show_wandb_run_name
                    )
                plot_pca(
                    test_data, combo['batch_size'], model, device, 
                    criterion, sorted_chem_names, all_embeddings_df,
                    ims_embeddings_df, 'Test', input_type, embedding_type, show_wandb_run_name
                    )
                break
        # if save_emb_pca_to_wandb:
        #     # true_embeddings, predicted_embeddings_flattened, chem_names = 
        #     preds_to_emb_pca_plot(predicted_embeddings, output_name_encodings, sorted_chem_names, embedding_df)

        # at last epoch print model architecture details (this will also show up in wandb log)
        print('-------------------------------------------')
        print('-------------------------------------------')
        print('Dataset: ', wandb_kwargs['dataset'])
        print('Target Embeddings: ', wandb_kwargs['target_embedding'])
        print('-------------------------------------------')
        print('-------------------------------------------')
        print(model)
        print('-------------------------------------------')
        print('-------------------------------------------')

        wandb.finish()

    print('Hyperparameters for best model: ')
    for key in best_hyperparams:
        print('   ', key, ' : ', best_hyperparams[key])
    
    return best_hyperparams

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------

def create_dataset_tensors(spectra_dataset, embedding_df, device, carl=False):
    """
    Create tensors from the provided spectra dataset and embedding DataFrame.

    Parameters:
    ----------
    spectra_dataset : pd.DataFrame
        DataFrame containing spectral data and chemical labels. Assumes specific 
        columns for processing based on the `carl` flag.

    embedding_df : pd.DataFrame
        DataFrame containing embeddings for chemicals, with 'Embedding Floats' 
        column corresponding to ChemNet embeddings.

    device : torch.device
        The device (CPU or GPU) on which to store the tensors.

    carl : bool, optional
        If True, processes the dataset assuming it has a different structure 
        (specifically without an 'Unnamed: 0' column). Default is False.

    Returns:
    -------
    tuple
        A tuple containing:
        - embeddings_tensor (torch.Tensor): Tensor of true embeddings for the chemicals.
        - spectra_tensor (torch.Tensor): Tensor of spectral data.
        - chem_encodings_tensor (torch.Tensor): Tensor of chemical name encodings.
        - spectra_indices_tensor (torch.Tensor): Tensor of indices corresponding to the spectra.
    """
    # drop first two cols ('Unnamed:0' and 'index') and last 9 cols ('Label' and OneHot encodings) to get just spectra
    if carl: # carl dataset has no 'Unnamed: 0' column
        spectra = spectra_dataset.iloc[:,1:-9]
        # embeddings_tensor = torch.Tensor([embedding_df['Embedding Floats'][chem_name] for chem_name in chem_labels]).to(device)
    else:
        spectra = spectra_dataset.iloc[:,2:-9]
        # embeddings_tensor = torch.Tensor([embedding_df['Embedding Floats'][chem_name] for chem_name in chem_labels]).to(device)
        
    chem_encodings = spectra_dataset.iloc[:,-8:]

    # create tensors of spectra, true embeddings, and chemical name encodings for train and val
    chem_labels = list(spectra_dataset['Label'])
    embeddings_tensor = torch.Tensor([embedding_df['Embedding Floats'][chem_name] for chem_name in chem_labels]).to(device)
    spectra_tensor = torch.Tensor(spectra.values).to(device)
    chem_encodings_tensor = torch.Tensor(chem_encodings.values).to(device)
    spectra_indices_tensor = torch.Tensor(spectra_dataset['index'].to_numpy()).to(device)

    return embeddings_tensor, spectra_tensor, chem_encodings_tensor, spectra_indices_tensor

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------

def create_dataset_tensors_for_generator(carl_dataset, embedding_preds, device=None, multiple_carls_per_spec=False):
    """
    Create tensors from the provided CARL dataset and embedding DataFrame.

    Parameters:
    ----------
    spectra_dataset : pd.DataFrame
        DataFrame containing CARL data and chemical labels. Assumes specific 
        columns for processing based on the `carl` flag.

    embedding_df : pd.DataFrame
        DataFrame containing encoder predicted embeddings.

    device : torch.device
        The device (CPU or GPU) on which to store the tensors.

    Returns:
    -------
    tuple
        A tuple containing:
        - embeddings_tensor (torch.Tensor): Tensor of true embeddings for the chemicals.
        - spectra_tensor (torch.Tensor): Tensor of spectral data.
        - chem_encodings_tensor (torch.Tensor): Tensor of chemical name encodings.
        - spectra_indices_tensor (torch.Tensor): Tensor of indices corresponding to the CARLS.
    """
    # drop first col ('index') and last 9 cols ('Label', OneHot encodings) to get just CARLS and predicted embeddings
    carls = carl_dataset.iloc[:,1:-9]
    # embeddings df doesn't have 'Label' col, so dropping last 8 cols instead of last 9
    embedding_preds = embedding_preds.iloc[:,1:-8]

    chem_encodings = carl_dataset.iloc[:,-8:]
    # del carl_dataset

    if device is not None:
        embeddings_preds = torch.Tensor(embedding_preds.values).to(device)
        carls = torch.Tensor(carls.values).to(device)
        chem_encodings = torch.Tensor(chem_encodings.values).to(device)
    else:
        embeddings_preds = torch.Tensor(embedding_preds.values)
        carls = torch.Tensor(carls.values)
        chem_encodings = torch.Tensor(chem_encodings.values)

    if multiple_carls_per_spec:
        # torch.Tensor changes the vals after decimal but I need those to stay the same so using torch.tensor instead
        carl_indices = torch.tensor(carl_dataset['index']).to(device)
    else:
        carl_indices = torch.Tensor(carl_dataset['index'].values).to(device)

    return embeddings_preds, carls, chem_encodings, carl_indices

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
 
def create_dataset_tensors_with_dask(spectra_file, embedding_df, device, carl=False):
    """
    Create tensors from the provided spectra dataset and embedding DataFrame using Dask.

    Parameters:
    ----------
    spectra_file : str
        Path to the CSV file containing spectral data and chemical labels.

    embedding_df : pd.DataFrame
        DataFrame containing embeddings for chemicals.

    device : torch.device
        The device (CPU or GPU) on which to store the tensors.

    carl : bool, optional
        If True, processes the dataset assuming it has a different structure.

    Returns:
    -------
    tuple
        A tuple containing:
        - embeddings_tensor (torch.Tensor)
        - spectra_tensor (torch.Tensor)
        - chem_encodings_tensor (torch.Tensor)
        - spectra_indices_tensor (torch.Tensor)
    """
    # Load the dataset as a Dask DataFrame
    spectra_dd = dd.read_csv(spectra_file)

    # Compute the necessary tensors
    if carl:
        spectra = spectra_dd.iloc[:, 1:-9]
    else:
        spectra = spectra_dd.iloc[:, 2:-9]

    chem_labels = spectra_dd['Label'].compute().tolist()
    embeddings = [embedding_df['Embedding Floats'][chem_name] for chem_name in chem_labels]

    # Create tensors directly from Dask DataFrame
    spectra_tensor = torch.tensor(spectra.compute().values, dtype=torch.float32).to(device)
    chem_encodings = spectra_dd.iloc[:, -8:].compute()
    chem_encodings_tensor = torch.tensor(chem_encodings.values, dtype=torch.float32).to(device)
    spectra_indices_tensor = torch.tensor(spectra_dd['index'].compute().values, dtype=torch.float32).to(device)

    # Convert embeddings to tensor
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)

    return embeddings_tensor, spectra_tensor, chem_encodings_tensor, spectra_indices_tensor

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
 
def create_dataset_tensors_from_chunks(spectra_dataset, embedding_df, device, chunk_size=None, carl=False):
    """
    Create tensors from the provided spectra dataset and embedding DataFrame.

    Parameters:
    ----------
    spectra_dataset : pd.DataFrame
        DataFrame containing spectral data and chemical labels. Assumes specific 
        columns for processing based on the `carl` flag.

    embedding_df : pd.DataFrame
        DataFrame containing embeddings for chemicals, with 'Embedding Floats' 
        column corresponding to chemical names.

    device : torch.device
        The device (CPU or GPU) on which to store the tensors.

    carl : bool, optional
        If True, processes the dataset assuming it has a different structure 
        (specifically without an 'Unnamed: 0' column). Default is False.

    Returns:
    -------
    tuple
        A tuple containing:
        - embeddings_tensor (torch.Tensor): Tensor of true embeddings for the chemicals.
        - spectra_tensor (torch.Tensor): Tensor of spectral data.
        - chem_encodings_tensor (torch.Tensor): Tensor of chemical name encodings.
        - spectra_indices_tensor (torch.Tensor): Tensor of indices corresponding to the spectra.
    """
    embeddings_list = []
    spectra_list = []
    chem_encodings_list = []
    indices_list = []

    # Process the dataset in chunks
    for chunk in pd.read_csv(spectra_dataset, chunksize=chunk_size):
        if carl:
            spectra = chunk.iloc[:, 1:-9]
            chem_labels = list(chunk['Label'])
            embeddings = [embedding_df['Embedding Floats'][chem_name] for chem_name in chem_labels]
        else:
            spectra = chunk.iloc[:, 2:-9]
            chem_labels = list(chunk['Label'])
            embeddings = [embedding_df['Embedding Floats'][chem_name] for chem_name in chem_labels]

        # Convert to tensors
        embeddings_tensor = torch.Tensor(embeddings).to(device)
        spectra_tensor = torch.Tensor(spectra.values).to(device)
        chem_encodings = chunk.iloc[:, -8:]
        chem_encodings_tensor = torch.Tensor(chem_encodings.values).to(device)
        spectra_indices_tensor = torch.Tensor(chunk['index'].to_numpy()).to(device)

        # Append to lists
        embeddings_list.append(embeddings_tensor)
        spectra_list.append(spectra_tensor)
        chem_encodings_list.append(chem_encodings_tensor)
        indices_list.append(spectra_indices_tensor)

    # Concatenate all tensors
    embeddings_tensor = torch.cat(embeddings_list).to(device)
    spectra_tensor = torch.cat(spectra_list).to(device)
    chem_encodings_tensor = torch.cat(chem_encodings_list).to(device)
    spectra_indices_tensor = torch.cat(indices_list).to(device)

    return embeddings_tensor, spectra_tensor, chem_encodings_tensor, spectra_indices_tensor


# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
 
class Generator(nn.Module):
  def __init__(self):
    super().__init__()
    self.encoder = nn.Sequential(
      nn.Linear(512,652),
      nn.LeakyReLU(inplace=True),
      nn.Linear(652,780),
      nn.LeakyReLU(inplace=True),
      nn.Linear(780, 908),
      nn.LeakyReLU(inplace=True),
      nn.Linear(908, 1036),
      nn.LeakyReLU(inplace=True),
      nn.Linear(1036, 1164),
      nn.LeakyReLU(inplace=True),
      nn.Linear(1164, 1292),
      nn.LeakyReLU(inplace=True),
      nn.Linear(1292, 1420),
      nn.LeakyReLU(inplace=True),
      nn.Linear(1420, 1548),
      nn.LeakyReLU(inplace=True),
      nn.Linear(1548, 1676),
    )

  def forward(self, x):
    x = self.encoder(x)
    return x

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
   
def set_up_gpu():
    if torch.cuda.is_available():
        # Get the list of GPUs
        gpus = GPUtil.getGPUs()

        # Find the GPU with the most free memory
        best_gpu = max(gpus, key=lambda gpu: gpu.memoryFree)

        # Print details about the selected GPU
        print(f"Selected GPU ID: {best_gpu.id}")
        print(f"  Name: {best_gpu.name}")
        print(f"  Memory Free: {best_gpu.memoryFree} MB")
        print(f"  Memory Used: {best_gpu.memoryUsed} MB")
        print(f"  GPU Load: {best_gpu.load * 100:.2f}%")

        # Set the device for later use
        device = torch.device(f'cuda:{best_gpu.id}')
        print('Current device ID: ', device)

        # Set the current device in PyTorch
        torch.cuda.set_device(best_gpu.id)
    else:
        device = torch.device('cpu')
        print('Using CPU')

    # Confirm the currently selected device in PyTorch
    print("PyTorch current device ID:", torch.cuda.current_device())
    print("PyTorch current device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

    return device

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------

def plot_and_save_generator_results(
    data, batch_size, sorted_chem_names, model, device, 
    criterion, num_plots, plot_overlap_pca=False, 
    save_plots_to_wandb=True, show_wandb_run_name=True, test_or_train='Train'
    ):
    # get predictions from trained model and plot them
    dataset = DataLoader(data, batch_size=batch_size)
    predicted_carls, output_name_encodings, _, _ = predict_embeddings(dataset, model, device, criterion)
    
    for _ in range(num_plots):
        random_carl = random.randint(0, len(data))
        encodings_list = [enc for enc_list in output_name_encodings for enc in enc_list]
        predicted_carls_list = [pred for pred_list in predicted_carls for pred in pred_list]       
        chem = sorted_chem_names[list(encodings_list[random_carl]).index(1)]
        plot_carl_real_synthetic_comparison(
            data[random_carl][2].cpu(), predicted_carls_list[random_carl], test_or_train, 
            chem, save_plots_to_wandb, show_wandb_run_name)

    if plot_overlap_pca:
        true_spectra = [spec[2].cpu() for spec in data]
        true_spectra_df = pd.DataFrame(true_spectra)
        spectra_labels = [sorted_chem_names[list(enc).index(1)] for enc in encodings_list]
        true_spectra_df['Labels'] = spectra_labels

        # synthetic_spectra_df = pd.DataFrame(predicted_carls_list)
        # print(synthetic_spectra_df.shape)
        # synthetic_spectra_df['Labels'] = spectra_labels
        # print('got here')
        # plot_generation_results_pca(
        #     true_spectra_df, synthetic_spectra_df.sample(n=1000, random_state=42), 
        #     sorted_chem_names, test_or_train, sample_size=1000, log_wandb=True,
        #     mse_insert=None, insert_position=[0.05, 0.05], show_wandb_run_name=True)
    

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------

def train_generator(
        train_data, val_data, test_data, device, config, wandb_kwargs, 
        model_hyperparams, sorted_chem_names, generator_path, 
        save_plots_to_wandb = True, early_stop_threshold=10, 
        show_wandb_run_name=True, lr_scheduler = True, 
        num_plots = 1, patience=5, plot_overlap_pca=False
        ):
    wandb.finish()
    # loss to compare for each model. Starting at infinity so it will be replaced by first model's first epoch loss 
    lowest_val_loss = np.inf

    keys = model_hyperparams.keys()
    values = model_hyperparams.values()

    # Generate all parameter combinations from model_config using itertools.product
    combinations = itertools.product(*values)


    # Iterate through each parameter combination and run model 
    for combo in combinations:
        # creating different var for model loss to use for early stopping
        lowest_val_model_loss = np.inf
        
        model = Generator().to(device)

        epochs_without_validation_improvement = 0
        combo = dict(zip(keys, combo))

        train_dataset = DataLoader(train_data, batch_size=combo['batch_size'], shuffle=True)
        val_dataset = DataLoader(val_data, batch_size=combo['batch_size'], shuffle=False)

        optimizer = torch.optim.AdamW(model.parameters(), lr = combo['learning_rate'])
        criterion = nn.MSELoss()

        final_lr = combo['learning_rate']

        if lr_scheduler:
            # Initialize the learning rate scheduler with patience of 5 epochs 
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=patience, factor=0.1, verbose=True)

        wandb_kwargs = update_wandb_kwargs(wandb_kwargs, combo)

        run_with_wandb(config, **wandb_kwargs)

        print('--------------------------')
        print('--------------------------')
        print('New run with hyperparameters:')
        for key in combo:
            print(key, ' : ', combo[key])

        for epoch in range(combo['epochs']):
            if epochs_without_validation_improvement < early_stop_threshold:
                model.train(True)

                # do a pass over the data
                # at last epoch get predicted embeddings and chem names
                if (epoch + 1) == combo['epochs']:
                    average_loss, _, _ = train_one_epoch(
                    train_dataset, device, model, criterion, optimizer, epoch, combo
                    )
                    wandb.log({'Learning Rate at Final Epoch':final_lr})
                    # save output plots to weights and biases
                    if save_plots_to_wandb:
                        plot_and_save_generator_results(
                            train_data, combo['batch_size'], sorted_chem_names, 
                            model, device, criterion, num_plots, plot_overlap_pca=plot_overlap_pca, 
                            save_plots_to_wandb=True, show_wandb_run_name=show_wandb_run_name,
                            test_or_train='Train'
                            )
                        plot_and_save_generator_results(
                            test_data, combo['batch_size'], sorted_chem_names, 
                            model, device, criterion, num_plots, plot_overlap_pca=False, 
                            save_plots_to_wandb=True, show_wandb_run_name=show_wandb_run_name,
                            test_or_train='Train'
                            )
            
                else:
                    average_loss = train_one_epoch(
                    train_dataset, device, model, criterion, optimizer, epoch, combo
                    )

                epoch_val_loss = 0  
                # evaluate model on validation data
                model.eval() # Set model to evaluation mode
                with torch.no_grad():
                    for val_batch, val_name_encodings, val_true_embeddings, _ in val_dataset:
                        val_batch = val_batch.to(device)
                        val_name_encodings = val_name_encodings.to(device)
                        val_true_embeddings = val_true_embeddings.to(device)

                        val_batch_predicted_embeddings = model(val_batch)

                        val_loss = criterion(val_batch_predicted_embeddings, val_true_embeddings)
                        # accumulate epoch validation loss
                        epoch_val_loss += val_loss.item()

                # divide by number of batches to calculate average loss
                val_average_loss = epoch_val_loss/len(val_dataset)

                if lr_scheduler:
                    scheduler.step(val_average_loss)  # Pass the validation loss to the scheduler
                    # get the new learning rate (to give to wandb)
                    final_lr = optimizer.param_groups[0]['lr']

                if val_average_loss < lowest_val_model_loss:
                    # check if val loss is improving for this model
                    epochs_without_validation_improvement = 0
                    lowest_val_model_loss = val_average_loss
                    # best_epoch = epoch + 1  # Store the best epoch

                    if val_average_loss < lowest_val_loss:
                        # if current epoch of current model is best performing (of all epochs and models so far), save model state
                        # Save the model
                        # torch.save(model.state_dict(), generator_path)
                        torch.save(model, generator_path)
                        print(f'Saved best model at epoch {epoch+1}')
                        lowest_val_loss = val_average_loss
                        best_hyperparams = combo
                    else:
                        print(f'Model best validation loss at {epoch+1}')
                
                else:
                    epochs_without_validation_improvement += 1

                # log losses and memory stats to wandb
                memory_info = psutil.virtual_memory()
                wandb.log({
                    "Generator Training Loss": average_loss, "Generator Validation Loss": val_average_loss, 
                    "memory_used": memory_info.used, "memory_percent": memory_info.percent
                    })

                if (epoch + 1) % 10 == 0 or epoch == 0:
                    print('Epoch[{}/{}]:'.format(epoch+1, combo['epochs']))
                    print(f'   Training loss: {average_loss}')
                    print(f'   Validation loss: {val_average_loss}')
                    print('-------------------------------------------')
    
            else:
                print(f'Validation loss has not improved in {epochs_without_validation_improvement} epochs. Stopping training at epoch {epoch+1}.')
                wandb.log({'Early Stopping Ecoch':epoch+1})
                wandb.log({'Learning Rate at Final Epoch':final_lr})
                plot_and_save_generator_results(
                    train_data, combo['batch_size'], sorted_chem_names, 
                    model, device, criterion, num_plots, plot_overlap_pca=plot_overlap_pca, 
                    save_plots_to_wandb=True, show_wandb_run_name=show_wandb_run_name,
                    test_or_train='Train'
                    )
                plot_and_save_generator_results(
                    test_data, combo['batch_size'], sorted_chem_names, 
                    model, device, criterion, num_plots, plot_overlap_pca=False, 
                    save_plots_to_wandb=True, show_wandb_run_name=show_wandb_run_name,
                    test_or_train='Train'
                    )
                # plot_and_save_generator_results(
                #             test_data, combo['batch_size'], sorted_chem_names, 
                #             model, device, criterion, num_plots, plot_overlap_pca=False, 
                #             save_plots_to_wandb=True, show_wandb_run_name=show_wandb_run_name
                #             )
                # train_dataset = DataLoader(train_data, batch_size=combo['batch_size'])
                # train_predicted_carls, train_output_name_encodings, _, _ = predict_embeddings(train_dataset, model, device, criterion)
                # test_dataset = DataLoader(test_data, batch_size=combo['batch_size'])
                # test_predicted_carls, test_output_name_encodings, _, _ = predict_embeddings(test_dataset, model, device, criterion)
                
                # for _ in range(num_plots):
                #     random_carl = random.randint(0, len(test_data))
                #     train_encodings_list = [enc for enc_list in train_output_name_encodings for enc in enc_list]
                #     test_encodings_list = [enc for enc_list in test_output_name_encodings for enc in enc_list]
                #     train_predicted_carls_list = [pred for pred_list in train_predicted_carls for pred in pred_list]
                #     test_predicted_carls_list = [pred for pred_list in test_predicted_carls for pred in pred_list]
                #     train_chem = sorted_chem_names[list(train_encodings_list[random_carl]).index(1)]
                #     test_chem = sorted_chem_names[list(test_encodings_list[random_carl]).index(1)]
                #     plot_carl_real_synthetic_comparison(train_data[random_carl][2].cpu(), train_predicted_carls_list[random_carl], 'Train', train_chem)
                #     plot_carl_real_synthetic_comparison(test_data[random_carl][2].cpu(), test_predicted_carls_list[random_carl], 'Test', test_chem)
                
                break

        # at last epoch print model architecture details (this will also show up in wandb log)
        print('-------------------------------------------')
        print('-------------------------------------------')
        print('Dataset: ', wandb_kwargs['dataset'])
        print('Target Embeddings: ', wandb_kwargs['target_embedding'])
        print('-------------------------------------------')
        print('-------------------------------------------')
        print(model)
        print('-------------------------------------------')
        print('-------------------------------------------')

        wandb.finish()

    print('Hyperparameters for best model: ')
    for key in best_hyperparams:
        print('   ', key, ' : ', best_hyperparams[key])
    
    return best_hyperparams

# ------------------------------------------------------------------------------------------    
# ------------------------------------------------------------------------------------------    
# ------------------------------------------------------------------------------------------    

def predict_carls(dataset, model, device, criterion):
    total_loss = 0

    model.eval() # Set model to evaluation mode
    predicted_carls = []
    output_name_encodings = []
    input_carl_indices = []

    with torch.no_grad():
        for batch, name_encodings, true_carls, carl_indices in dataset:
            batch = batch.to(device)
            true_carls = true_carls.to(device)

            batch_predicted_carls = model(batch)
            predicted_carls.append(batch_predicted_carls)
            output_name_encodings.append(name_encodings)
            input_carl_indices.append(carl_indices)

            # print(batch_predicted_embeddings.shape, true_embeddings.shape)

            loss = criterion(batch_predicted_carls, true_carls)
            # accumulate loss
            total_loss += loss.item()

    # divide by number of batches to calculate average loss
    average_loss = total_loss/len(dataset)
    return predicted_carls, output_name_encodings, average_loss, input_carl_indices

# ------------------------------------------------------------------------------------------    
# ------------------------------------------------------------------------------------------    
# ------------------------------------------------------------------------------------------  

def generate_average_bkg_spectrum(bkg_df, n, num_avg_spectra=1, random_state=42):
    avg_spectra = []
    for _ in range(num_avg_spectra):
        bkg_sample = bkg_df.sample(n=n, random_state=random_state).iloc[:,1:-1]
        bkg_df = bkg_df.drop(bkg_sample.index)
        bkg_sample.reset_index(inplace=True)
        bkg_sample.drop(columns=['index'], inplace=True)

        avg_spectrum = bkg_sample.mean()
        avg_spectra.append(avg_spectrum)

    return avg_spectra