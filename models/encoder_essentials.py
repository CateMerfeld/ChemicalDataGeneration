#%%
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import functions as f
#%%
class Encoder(nn.Module):
    def __init__(self, input_size, output_size, num_layers):
        super().__init__()
        layers = []
        layer_sizes = np.linspace(input_size, output_size, num_layers + 1, dtype=int)
        for i in range(num_layers):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < num_layers - 1:
                layers.append(nn.LeakyReLU(inplace=True))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)

def train_model(model, train_data, val_data, epochs, batch_size, learning_rate, criterion, device):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch, name_encodings, true_embeddings, _ in train_data:
            batch = batch.to(device)
            name_encodings = name_encodings.to(device)
            true_embeddings = true_embeddings.to(device)

            optimizer.zero_grad()
            batch_predicted_embeddings = model(batch)
            loss = criterion(batch_predicted_embeddings, true_embeddings)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch[0].to(device), batch[1].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

    return model
#%%
train_data = pd.read_feather('../../scratch/val_data.feather')
val_data = pd.read_feather('../../scratch/test_data.feather')
name_smiles_embedding_df_file_path = '../../scratch/name_smiles_embedding_file.csv'
device = f.set_up_gpu()
name_smiles_embedding_df = f.format_embedding_df(name_smiles_embedding_df_file_path)


y_train, x_train, train_chem_encodings_tensor, train_indices_tensor = f.create_dataset_tensors(
    train_data, name_smiles_embedding_df, device, start_idx=2, stop_idx=-9)
sorted_chem_names = list(train_data.columns[-8:])
del train_data

y_val, x_val, val_chem_encodings_tensor, val_indices_tensor = f.create_dataset_tensors(
    val_data, name_smiles_embedding_df, device, start_idx=2, stop_idx=-9)
del val_data

train_data = TensorDataset(x_train, train_chem_encodings_tensor, y_train, train_indices_tensor)
val_data = TensorDataset(x_val, val_chem_encodings_tensor, y_val, val_indices_tensor)
#%%
encoder = Encoder(input_size=x_train.shape[1], output_size=512, num_layers=3).to(device)

#%%
model = train_model(
    model=encoder,
    train_data=train_data,
    val_data=val_data,
    epochs=10,
    batch_size=64,
    learning_rate=0.001,
    criterion=nn.MSELoss(),
    device=device
)