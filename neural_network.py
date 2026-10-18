import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import itertools

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PyTorchMLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes):
        super(PyTorchMLP, self).__init__()
        layers = []
        in_dim = input_dim
        for h_size in hidden_sizes:
            layers.append(nn.Linear(in_dim, h_size))
            layers.append(nn.ReLU())
            in_dim = h_size
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def train_optimized_mlp(X_train, y_train, X_val, y_val, hidden_sizes=(128, 64, 32), target_scale=1000.0, epochs=100, batch_size=64, lr=0.001):
    # Extract numpy array elements to prevent pandas Series conversion issues with PyTorch
    y_tr_np = y_train.values if hasattr(y_train, 'values') else np.array(y_train)
    X_tr_np = X_train.values if hasattr(X_train, 'values') else np.array(X_train)
    
    X_tr_t = torch.tensor(X_tr_np, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr_np * target_scale, dtype=torch.float32).unsqueeze(1)
    
    dataset = TensorDataset(X_tr_t, y_tr_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = PyTorchMLP(X_tr_np.shape[1], hidden_sizes).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    loss_history = []
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)
        loss_history.append(epoch_loss / len(dataset))
    return model, loss_history

def evaluate_pytorch_model(model, X, target_scale=1000.0):
    X_np = X.values if hasattr(X, 'values') else np.array(X)
    X_t = torch.tensor(X_np, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        preds = model(X_t).cpu().numpy().flatten()
    return preds / target_scale

def run_parameter_sweep(X_train, y_train, X_val, y_val, target_scale=1000.0):
    layer_options = [1, 2, 3]
    node_options = [32, 64, 128]
    architectures = []
    for num_layers in layer_options:
        for arch in itertools.product(node_options, repeat=num_layers):
            architectures.append(arch)
    results = []
    for arch in architectures:
        model, _ = train_optimized_mlp(X_train, y_train, X_val, y_val, hidden_sizes=arch, target_scale=target_scale, epochs=10, batch_size=64, lr=0.001)
        train_preds = evaluate_pytorch_model(model, X_train, target_scale)
        train_mse = mean_squared_error(y_train, train_preds)
        val_preds = evaluate_pytorch_model(model, X_val, target_scale)
        val_mse = mean_squared_error(y_val, val_preds)
        results.append({
            'architecture': arch,
            'train_mse': train_mse,
            'val_mse': val_mse
        })
    results_df = pd.DataFrame(results)
    results_df['Arch_Str'] = results_df['architecture'].apply(lambda x: str(x))
    return results_df
