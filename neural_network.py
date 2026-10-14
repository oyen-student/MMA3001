
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import itertools
import pandas as pd

def train_optimized_mlp(X_train, y_train, X_val, y_val, target_scale=1000.0):
    y_train_scaled = y_train * target_scale
    
    mlp = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size=64,
        learning_rate_init=0.001,
        learning_rate='adaptive',
        max_iter=150,
        tol=1e-6,
        n_iter_no_change=25,
        random_state=42,
        verbose=False
    )
    
    print("Training standard deep MLP...")
    mlp.fit(X_train, y_train_scaled)
    return mlp

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
    y_train_scaled = y_train * target_scale
    layer_options = [1, 2, 3]
    node_options = [32, 64, 128]
    
    architectures = []
    for num_layers in layer_options:
        for arch in itertools.product(node_options, repeat=num_layers):
            architectures.append(arch)
            
    results = []
    for arch in architectures:
        mlp = MLPRegressor(
            hidden_layer_sizes=arch,
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size=64,
            learning_rate_init=0.001,
            learning_rate='adaptive',
            max_iter=100,
            tol=1e-5,
            random_state=42
        )
        mlp.fit(X_train, y_train_scaled)
        
        train_preds = mlp.predict(X_train) / target_scale
        train_mse = mean_squared_error(y_train, train_preds)
        
        val_preds = mlp.predict(X_val) / target_scale
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
    plt.title('MLP Model Parameter Sweep: Train vs Validation Error Comparison', fontsize=14, fontweight='bold')
    plt.grid(True, which='both', linestyle='--', alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()
