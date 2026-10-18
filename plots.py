import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna

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

def plot_loss_convergence(loss_history):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(loss_history) + 1), loss_history, color='teal', linewidth=2.5)
    plt.title('MLP Neural Network Training Loss Convergence', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch (Iteration)')
    plt.ylabel('Scaled MSE Loss')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

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
    plt.title('MLP Parameter Sweep: Train vs Validation Error', fontsize=14, fontweight='bold')
    plt.grid(True, which='both', linestyle='--', alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_optuna_convergence(study):
    trials = study.trials
    trial_numbers = [t.number for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    validation_mses = [t.value for t in trials if t.state == optuna.trial.TrialState.COMPLETE]

    running_minimums = []
    current_min = float('inf')
    for val in validation_mses:
        if val < current_min:
            current_min = val
        running_minimums.append(current_min)

    plt.figure(figsize=(12, 5))
    plt.plot(trial_numbers, validation_mses, marker='o', linestyle='--', color='tomato', alpha=0.7, label='Trial Objective (Validation MSE)')
    plt.step(trial_numbers, running_minimums, where='post', color='darkblue', linewidth=2.5, label='Best Value Found')

    plt.yscale('log')
    plt.xlabel('Trial Number', fontweight='bold')
    plt.ylabel('Validation MSE (Log Scale)', fontweight='bold')
    plt.title('Optuna Bayesian Optimization Convergence History', fontsize=13, fontweight='bold')
    plt.grid(True, which="both", linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()
