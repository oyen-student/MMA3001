
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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

def train_optimized_mlp(X_train, y_train, X_val, y_val, hidden_sizes=(128, 64, 32), target_scale=1000.0, epochs=150, batch_size=64, lr=0.001):
    # Prepare inputs and targets scaled to Tensor
    X_tr_t = torch.tensor(X_train, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train * target_scale, dtype=torch.float32).unsqueeze(1)
    
    dataset = TensorDataset(X_tr_t, y_tr_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = PyTorchMLP(X_train.shape[1], hidden_sizes).to(device)
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
        loss_history.append(epoch_loss / len(X_train))
        
    return model, loss_history

def evaluate_pytorch_model(model, X_data, target_scale=1000.0):
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_data, dtype=torch.float32).to(device)
        preds = model(X_t).cpu().numpy().flatten()
    return preds / target_scale

def plot_predictions(y_true, y_pred, title):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.4, color='darkslateblue', s=15)
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.title(title, fontsize=12, fontweight='bold')
    plt.xlabel('Actual C_w')
    plt.ylabel('Predicted C_w')
    plt.grid(True, linestyle='--', alpha=0.5)
    text_str = f'MSE: {mse:.4e}\nMAE: {mae:.4e}\nR²: {r2:.4f}'
    plt.gca().text(0.05, 0.95, text_str, transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85))
    plt.tight_layout()
    plt.show()

def run_parameter_sweep(X_train, y_train, X_val, y_val, target_scale=1000.0):
    layer_options = [1, 2, 3]
    node_options = [32, 64, 128]

    architectures = []
    for num_layers in layer_options:
        for arch in itertools.product(node_options, repeat=num_layers):
            architectures.append(arch)

    results = []
    for arch in architectures:
        model, _ = train_optimized_mlp(X_train, y_train, X_val, y_val, hidden_sizes=arch, target_scale=target_scale, epochs=80, batch_size=64, lr=0.001)
        
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

def plot_sweep_results(results_df):
    plt.figure(figsize=(16, 7))
    x_indices = np.arange(len(results_df))
    bar_width = 0.35

    plt.bar(x_indices - bar_width/2, results_df['train_mse'], width=bar_width, label='Training MSE', color='royalblue', alpha=0.85)
    plt.bar(x_indices + bar_width/2, results_df['val_mse'], width=bar_width, label='Validation MSE', color='darkorange', alpha=0.85)

    plt.xticks(x_indices, results_df['Arch_Str'], rotation=90, fontsize=9)
    plt.yscale('log')
    plt.xlabel('Hidden Layer Architecture (Layers, Nodes)', fontweight='bold')
    plt.ylabel('Mean Squared Error (Log Scale)', fontweight='bold')
    plt.title('PyTorch MLP Parameter Sweep: Train vs Validation Error', fontsize=14, fontweight='bold')
    plt.grid(True, which='both', linestyle='--', alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()
