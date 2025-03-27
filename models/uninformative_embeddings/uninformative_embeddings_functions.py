from collections import OrderedDict
import torch.nn as nn
import itertools
import numpy as np
import torch
from torch.utils.data import DataLoader
import wandb


import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import functions as f
import plotting_functions as pf

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
class Encoder(nn.Module):
  def __init__(self, input_size=1676, output_size=8, n_layers=3):
    super().__init__()

    layers = OrderedDict()
    if n_layers > 1:
        size_reduction_per_layer = (input_size - output_size)/n_layers
        for i in range(n_layers-1):
            layer_input_size = input_size - int(size_reduction_per_layer)*i
            layer_output_size = input_size - int(size_reduction_per_layer)*(i+1)
            layers[f'fc{i}'] = nn.Linear(layer_input_size, layer_output_size)
            layers[f'relu{i}'] = nn.LeakyReLU(inplace=True)

        layers['final'] = nn.Linear(layer_output_size, output_size)

    else:
        layers['final'] = nn.Linear(input_size, output_size)

    self.encoder = nn.Sequential(layers)

  def forward(self, x):
    x = self.encoder(x)
    return x
  
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
def train_model(
    model_type, base_model, train_data, val_data, test_data, device, config, wandb_kwargs,
    all_embeddings_df, ims_embeddings_df, model_hyperparams, sorted_chem_names, 
    encoder_path, criterion, input_type, embedding_type, show_wandb_run_name=True,
    save_emb_pca_to_wandb = True, early_stop_threshold=np.inf, 
    lr_scheduler=False, patience=5,
    ):

    # loss to compare for each model. Starting at infinity so it will be replaced by first model's first epoch loss 
    lowest_val_loss = np.inf

    keys = model_hyperparams.keys()
    values = model_hyperparams.values()

    # Generate all parameter combinations from model_config using itertools.product
    combinations = itertools.product(*values)

    # Iterate through each parameter combination and run model 
    for combo in combinations:
        model = base_model.to(device) 

        # creating different var for model loss to use for early stopping
        lowest_val_model_loss = np.inf           
        epochs_without_validation_improvement = 0

        combo = dict(zip(keys, combo))

        train_dataset = DataLoader(train_data, batch_size=combo['batch_size'], shuffle=True)
        val_dataset = DataLoader(val_data, batch_size=combo['batch_size'], shuffle=False)

        optimizer = torch.optim.AdamW(model.parameters(), lr = combo['learning_rate'])

        final_lr = combo['learning_rate']

        if lr_scheduler:
            # Initialize the learning rate scheduler with patience of 5 epochs 
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=patience, factor=0.1, verbose=True)

        wandb_kwargs = f.update_wandb_kwargs(wandb_kwargs, combo)

        f.run_with_wandb(config, **wandb_kwargs)

        print('--------------------------')
        print('--------------------------')
        print('New run with hyperparameters:')
        for key in combo:
            print(key, ' : ', combo[key])

        for epoch in range(1, combo['epochs']+1):
            if epochs_without_validation_improvement < early_stop_threshold:
                model.train(True)

                # do a pass over the data
                # at last epoch get predicted embeddings and chem names
                if epoch == combo['epochs']:
                    average_loss, _, _ = f.train_one_epoch(
                    train_dataset, device, model, criterion, 
                    optimizer, epoch, combo
                    )
                    wandb.log({'Learning Rate at Final Epoch':final_lr})
                    # save output pca to weights and biases
                    if save_emb_pca_to_wandb:
                        # plot_pca gets predictions from trained model and plots them
                        pf.plot_pca(
                            train_data, combo['batch_size'], model, device, 
                            criterion, sorted_chem_names, all_embeddings_df, 
                            ims_embeddings_df, 'Train', input_type, embedding_type, 
                            show_wandb_run_name
                            )
                        pf.plot_pca(
                            test_data, combo['batch_size'], model, device, 
                            criterion, sorted_chem_names, all_embeddings_df,
                            ims_embeddings_df, 'Test', input_type, embedding_type, 
                            show_wandb_run_name
                            )
                else:
                    average_loss = f.train_one_epoch(
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
                        print(f'Saved best model at epoch {epoch}')
                        lowest_val_loss = val_average_loss
                        best_hyperparams = combo
                    else:
                        print(f'Model best validation loss at {epoch}')
                
                else:
                    epochs_without_validation_improvement += 1

                # log losses to wandb
                wandb.log({f"{model_type} Training Loss": average_loss, f"{model_type} Validation Loss": val_average_loss})

                if epoch % 10 == 0 or epoch == 0:
                    print('Epoch[{}/{}]:'.format(epoch, combo['epochs']))
                    print(f'   Training loss: {average_loss}')
                    print(f'   Validation loss: {val_average_loss}')
                    print('-------------------------------------------')
            else:
                print(f'Validation loss has not improved in {epochs_without_validation_improvement} epochs. Stopping training at epoch {epoch}.')
                wandb.log({'Early Stopping Ecoch':epoch})
                wandb.log({'Learning Rate at Final Epoch':final_lr})

                pf.plot_pca(
                    train_data, combo['batch_size'], model, device, 
                    criterion, sorted_chem_names, all_embeddings_df, 
                    ims_embeddings_df, 'Train', input_type, embedding_type, 
                    show_wandb_run_name
                    )
                pf.plot_pca(
                    test_data, combo['batch_size'], model, device, 
                    criterion, sorted_chem_names, all_embeddings_df,
                    ims_embeddings_df, 'Test', input_type, embedding_type, 
                    show_wandb_run_name
                    )
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