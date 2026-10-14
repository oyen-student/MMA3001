import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def plot_predictions(y_true, y_pred, title):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.5, color='darkslateblue')
    mn, mx = min(y_true), max(y_true)
    plt.plot([mn, mx], [mn, mx], color='red', linestyle='--', lw=2)
    plt.title(title, fontsize=12, fontweight='bold')
    plt.xlabel('Actual C_w', fontweight='bold')
    plt.ylabel('Predicted C_w', fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

def plot_loss_convergence(loss_history):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, color='teal', lw=2, label='Training Loss')
    plt.title('Neural Network Loss Convergence History', fontsize=12, fontweight='bold')
    plt.xlabel('Epochs', fontweight='bold')
    plt.ylabel('Loss (MSE)', fontweight='bold')
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()

def plot_sweep_results(results_df):
    plt.figure(figsize=(12, 6))
    x = np.arange(len(results_df))
    width = 0.35
    plt.bar(x - width/2, results_df['train_mse'], width, label='Train MSE', color='royalblue')
    plt.bar(x + width/2, results_df['val_mse'], width, label='Val MSE', color='darkorange')
    plt.xticks(x, results_df['Arch_Str'], rotation=45)
    plt.yscale('log')
    plt.ylabel('MSE (Log Scale)', fontweight='bold')
    plt.title('Parameter Sweep Architectures vs MSE', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.3)
    plt.tight_layout()

def plot_optuna_convergence(study):
    plt.figure(figsize=(10, 5))
    trials = study.trials
    values = [t.value for t in trials if t.value is not None]
    plt.plot(values, marker='o', color='crimson', lw=2, label='Trial MSE')
    plt.title('Bayesian Optimization (Optuna) Convergence History', fontsize=12, fontweight='bold')
    plt.xlabel('Trial Iteration', fontweight='bold')
    plt.ylabel('Best Objective Value', fontweight='bold')
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
