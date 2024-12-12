import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import wandb
from sklearn.decomposition import PCA
import itertools

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


def train_one_epoch(train_dataset, device, model, criterion, optimizer, epoch, combo):
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

    # forward pass
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

    # Currently, preds and name encodings are lists of [n_batches, batch_size], flattening to lists of [n_samples]
    predicted_embeddings_flattened = [emb.cpu().detach().numpy() for emb_list in predicted_embeddings for emb in emb_list]
    chem_name_encodings_flattened = [enc.cpu() for enc_list in output_name_encodings for enc in enc_list]

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

def predict_embeddings(dataset, model, device, criterion):
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

            batch_predicted_embeddings = model(batch)
            predicted_embeddings.append(batch_predicted_embeddings)
            output_name_encodings.append(name_encodings)
            input_spectra_indices.append(spectra_indices)

            # print(batch_predicted_embeddings.shape, true_embeddings.shape)

            loss = criterion(batch_predicted_embeddings, true_embeddings)
            # accumulate loss
            total_loss += loss.item()

    # divide by number of batches to calculate average loss
    average_loss = total_loss/len(dataset)
    return predicted_embeddings, output_name_encodings, average_loss, input_spectra_indices

def plot_emb_pca(
        all_embeddings, ims_embeddings, input_type, embedding_type=None, mass_spec_embeddings = None, log_wandb=False, 
        chemnet_embeddings_to_plot=None, mse_insert=None, insert_position=[0.05, 0.05], show_wandb_run_name=True):
    """
    Perform PCA on chemical embeddings and plot the transformed data, including IMS and Mass Spec embeddings if provided.

    Parameters:
    ----------
    all_embeddings : pd.DataFrame
        A dataframe containing ChemNet embeddings for all chemicals. 
        Each column represents one chemical's ChemNet embedding.
    ims_embeddings : pd.DataFrame
        A dataframe containing IMS (ion mobility spectrometry) embeddings. Must include a 'Label' column
        with chemical names and additional columns for embedding features.
    mass_spec_embeddings : pd.DataFrame, optional
        A dataframe containing mass spectrometry embeddings. Similar structure to `ims_embeddings`.
        Default is None, meaning mass spec embeddings are not included.
    log_wandb : bool, optional
        If True, logs the generated plot to Weights and Biases (wandb). Default is True.
    chemnet_embeddings_to_plot : pd.DataFrame, optional
        A dataframe containing ChemNet embeddings for all chemicals TO BE PLOTTED. 
        Each column represents one chemical's ChemNet embedding.

    Returns:
    -------
    None
        Displays the PCA scatter plot with ChemNet, IMS, and Mass Spec embeddings. 
        Optionally logs the plot to wandb if `log_wandb` is True.

    Notes:
    -----
    - PCA is performed on the transpose of `all_embeddings` so that embeddings for ims and mass spec data can be plotted to the same space.
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
        # only label 1st 8 chemicals to avoid giant legend
        # ax.scatter(0,0, color = color, label=chem)
        if idx < 8:
            ax.scatter(transformed_embeddings[idx, 0], transformed_embeddings[idx, 1], color = color, label=chem)#, s=200)
        else:
            ax.scatter(transformed_embeddings[idx, 0], transformed_embeddings[idx, 1], color = color)#, s=75)
        # Transform ims_embeddings for the current chemical, if we have ims data for chem
        if chem in ims_labels:
            # transform all data for the given chemical. Exclude last col (label)
            ims_transformed = pca.transform(ims_embeddings[ims_embeddings['Label'] == chem].iloc[:, :-1])
            
            # Scatter plot for ims_embeddings with a different marker
            ax.scatter(ims_transformed[:, 0], ims_transformed[:, 1], marker='o', facecolors='none', edgecolors=color)#marker='x', color=color)#, s=75)
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
    plt.Line2D([0], [0], marker='o', color='w', label="IMS", markerfacecolor='none', markeredgecolor='black', markersize=6),
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
        plt.title(f'{embedding_type} vs. {input_type} Encoder Output PCA', fontsize=18)
    else:
        plt.title(f'ChemNet vs. {input_type} Encoder Output PCA', fontsize=18)

    if log_wandb:
        plt.savefig('tmp_plot.png', format='png', dpi=300)
        wandb.log({'PCA of Predicted Chemical Embeddings': wandb.Image('tmp_plot.png')})

    plt.show()

def plot_pca(
    data, batch_size, model, device, encoder_criterion, sorted_chem_names, 
    all_embeddings_df, ims_embeddings_df, 
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

    input_type : str
        The type of input data used for embeddings.

    embedding_type : str, optional
        The type of embedding being visualized.

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
        all_embeddings_df, preds_df, input_type=input_type, 
        embedding_type=embedding_type, log_wandb=log_wandb, 
        chemnet_embeddings_to_plot=true_embeddings, mse_insert=avg_loss,
        show_wandb_run_name=show_wandb_run_name
        )
    
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
      nn.Linear(652, 512),
    )

  def forward(self, x):
    x = self.encoder(x)
    return x
  
def train_model(
        model_type, train_data, val_data, test_data, device, config, wandb_kwargs, 
        all_embeddings_df, ims_embeddings_df, model_hyperparams, sorted_chem_names, 
        encoder_path, save_emb_pca_to_wandb = True, early_stop_threshold=10, 
        embedding_type='ChemNet', show_wandb_run_name=True
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

    embedding_type : str, optional
        The type of embedding being used (e.g., 'ChemNet'). Default is 'ChemNet'.

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
            encoder = Encoder().to(device)
        epochs_without_validation_improvement = 0
        combo = dict(zip(keys, combo))

        train_dataset = DataLoader(train_data, batch_size=combo['batch_size'], shuffle=True)
        val_dataset = DataLoader(val_data, batch_size=combo['batch_size'], shuffle=False)

        encoder_optimizer = torch.optim.AdamW(encoder.parameters(), lr = combo['learning_rate'])
        encoder_criterion = nn.MSELoss()

        wandb_kwargs = update_wandb_kwargs(wandb_kwargs, combo)

        run_with_wandb(config, **wandb_kwargs)

        print('--------------------------')
        print('--------------------------')
        print('New run with hyperparameters:')
        for key in combo:
            print(key, ' : ', combo[key])

        for epoch in range(combo['epochs']):
            if epochs_without_validation_improvement < early_stop_threshold:
                encoder.train(True)

                # do a pass over the data
                # at last epoch get predicted embeddings and chem names
                if (epoch + 1) == combo['epochs']:
                    average_loss, _, _ = train_one_epoch(
                    train_dataset, device, encoder, encoder_criterion, encoder_optimizer, epoch, combo
                    )
                    # save output pca to weights and biases
                    if save_emb_pca_to_wandb:
                        # plot_pca gets predictions from trained model and plots them
                        plot_pca(
                            train_data, combo['batch_size'], encoder, device, 
                            encoder_criterion, sorted_chem_names, all_embeddings_df, 
                            ims_embeddings_df, 'Train', embedding_type, show_wandb_run_name
                            )
                        plot_pca(
                            test_data, combo['batch_size'], encoder, device, 
                            encoder_criterion, sorted_chem_names, all_embeddings_df,
                            ims_embeddings_df, 'Test', embedding_type, show_wandb_run_name
                            )
                else:
                    average_loss = train_one_epoch(
                    train_dataset, device, encoder, encoder_criterion, encoder_optimizer, epoch, combo
                    )

                epoch_val_loss = 0  
                # evaluate model on validation data
                encoder.eval() # Set model to evaluation mode
                with torch.no_grad():
                    for val_batch, val_name_encodings, val_true_embeddings, _ in val_dataset:
                        val_batch = val_batch.to(device)
                        val_name_encodings = val_name_encodings.to(device)
                        val_true_embeddings = val_true_embeddings.to(device)

                        val_batch_predicted_embeddings = encoder(val_batch)

                        val_loss = encoder_criterion(val_batch_predicted_embeddings, val_true_embeddings)
                        # accumulate epoch validation loss
                        epoch_val_loss += val_loss.item()

                # divide by number of batches to calculate average loss
                val_average_loss = epoch_val_loss/len(val_dataset)

                if val_average_loss < lowest_val_model_loss:
                    # check if val loss is improving
                    epochs_without_validation_improvement = 0

                    if val_average_loss < lowest_val_loss:
                        # if current epoch of current model is best performing (of all epochs and models so far), save model state
                        best_epoch = epoch + 1  # Store the best epoch
                        # Save the model state
                        torch.save(encoder.state_dict(), encoder_path)
                        print(f'Saved best model at epoch {best_epoch}')
                        lowest_val_loss = val_average_loss
                        best_hyperparams = combo
                
                else:
                    epochs_without_validation_improvement += 1
                # log losses to wandb
                wandb.log({"Encoder Training Loss": average_loss, "Encoder Validation Loss": val_average_loss})

                if (epoch + 1) % 10 == 0:
                    print('Epoch[{}/{}]:'.format(epoch+1, combo['epochs']))
                    print(f'   Training loss: {average_loss}')
                    print(f'   Validation loss: {val_average_loss}')
                    print('-------------------------------------------')
            else:
                print(f'Validation loss has not improved in {epochs_without_validation_improvement} epochs. Stopping training at epoch {epoch}.')
                wandb.log({'Early Stopping Ecoch':epoch})
                plot_pca(
                    train_data, combo['batch_size'], encoder, device, 
                    encoder_criterion, sorted_chem_names, all_embeddings_df, 
                    ims_embeddings_df, 'Train', embedding_type, show_wandb_run_name
                    )
                plot_pca(
                    test_data, combo['batch_size'], encoder, device, 
                    encoder_criterion, sorted_chem_names, all_embeddings_df,
                    ims_embeddings_df, 'Test', embedding_type, show_wandb_run_name
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
        print(encoder)
        print('-------------------------------------------')
        print('-------------------------------------------')

        wandb.finish()

    print('Hyperparameters for best model: ')
    for key in best_hyperparams:
        print('   ', key, ' : ', best_hyperparams[key])
    
    return best_hyperparams

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
    # drop first two cols ('Unnamed:0' and 'index') and last 9 cols ('Label' and OneHot encodings) to get just spectra
    if carl: # carl dataset has no 'Unnamed: 0' column
        spectra = spectra_dataset.iloc[:,1:-9]
    else:
        spectra = spectra_dataset.iloc[:,2:-9]
    chem_encodings = spectra_dataset.iloc[:,-8:]

    # create tensors of spectra, true embeddings, and chemical name encodings for train and val
    chem_labels = list(spectra_dataset['Label'])
    embeddings_tensor = torch.Tensor([embedding_df['Embedding Floats'][chem_name] for chem_name in chem_labels]).to(device)
    spectra_tensor = torch.Tensor(spectra.values).to(device)
    chem_encodings_tensor = torch.Tensor(chem_encodings.values).to(device)
    spectra_indices_tensor = torch.Tensor(spectra_dataset['index'].to_numpy()).to(device)

    return embeddings_tensor, spectra_tensor, chem_encodings_tensor, spectra_indices_tensor
