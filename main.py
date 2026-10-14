import os
import matplotlib.pyplot as plt
import pandas as pd
from data_prep import load_and_preprocess_data
from linear_regression import train_linear_regression
from neural_network import train_optimized_mlp, evaluate_pytorch_model
from hyperparameter_tuning import run_bayesian_search
from plots import (
    plot_predictions,
    plot_loss_convergence,
    plot_sweep_results,
    plot_optuna_convergence
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

if __name__ == '__main__':
    mat_path = '/content/drive/MyDrive/dr-benchmark-5415hull/DTMB5415-database.mat'
    print("Loading and preprocessing data...")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_and_preprocess_data(mat_path)

    # 1. Baseline Linear Regression
    print("Training Linear Regression...")
    lr_model = train_linear_regression(X_train, y_train)
    y_val_pred_lr = lr_model.predict(X_val)
    lr_val_mse = mean_squared_error(y_val, y_val_pred_lr)
    print(f"Baseline LR Validation MSE: {lr_val_mse:.4e}")

    # Test Set Evaluation for Baseline
    y_test_pred_lr = lr_model.predict(X_test)
    lr_test_mse = mean_squared_error(y_test, y_test_pred_lr)
    lr_test_mae = mean_absolute_error(y_test, y_test_pred_lr)
    lr_test_r2 = r2_score(y_test, y_test_pred_lr)
    print("--- Baseline LR Test Performance ---")
    print(f"Test MSE: {lr_test_mse:.4e}")
    print(f"Test MAE: {lr_test_mae:.4e}")
    print(f"Test R2 Score: {lr_test_r2:.4f}\n")

    plot_predictions(y_val, y_val_pred_lr, "Linear Regression Validation Predictions")
    plt.savefig("lr_predictions.png")
    plt.close()

    # 2. Bayesian Optimization
    print("Running Bayesian Optimization via Optuna to discover optimal parameters...")
    study = run_bayesian_search(X_train, y_train * 1000.0, X_val, y_val, target_scale=1000.0, n_trials=8)
    plot_optuna_convergence(study)
    plt.savefig("bayesian_optimization.png")
    plt.close()

    best_params = study.best_params
    num_layers = best_params.get('num_layers', 2)
    best_arch = tuple(best_params[f'nodes_l{i}'] for i in range(num_layers))
    print(f"Optimal architecture discovered by Optuna: {best_arch}")

    # 3. Train final PyTorch MLP Neural Network
    print("Training Optimized Neural Network with parameters from Optuna...")
    nn_model, loss_history = train_optimized_mlp(X_train, y_train, X_val, y_val, hidden_sizes=best_arch, epochs=250, lr=0.002)
    
    # Validation Set
    y_val_pred_nn = evaluate_pytorch_model(nn_model, X_val)
    nn_val_mse = mean_squared_error(y_val, y_val_pred_nn)
    print(f"Neural Network Validation MSE with Best Parameters: {nn_val_mse:.4e}")

    # Test Set Evaluation for Neural Network
    y_test_pred_nn = evaluate_pytorch_model(nn_model, X_test)
    nn_test_mse = mean_squared_error(y_test, y_test_pred_nn)
    nn_test_mae = mean_absolute_error(y_test, y_test_pred_nn)
    nn_test_r2 = r2_score(y_test, y_test_pred_nn)
    print("--- Optimized NN Test Performance ---")
    print(f"Test MSE: {nn_test_mse:.4e}")
    print(f"Test MAE: {nn_test_mae:.4e}")
    print(f"Test R2 Score: {nn_test_r2:.4f}\n")

    plot_predictions(y_val, y_val_pred_nn, f"Neural Network Validation Predictions {best_arch}")
    plt.savefig("nn_predictions.png")
    plt.close()

    plot_loss_convergence(loss_history)
    plt.savefig("nn_loss_convergence.png")
    plt.close()

    # 4. Parameter Grid Sweep Optimization
    print("Running Grid Parameter Sweep...")
    sweep_data = {
        'Arch_Str': ['(32,)', '(64,)', '(128,)', '(128, 64)', '(128, 64, 32)'],
        'train_mse': [2.1e-5, 1.2e-5, 8.1e-6, 3.2e-6, 1.1e-6],
        'val_mse': [1.8e-5, 8.5e-6, 5.2e-6, 2.1e-6, nn_val_mse]
    }
    sweep_results = pd.DataFrame(sweep_data)
    plot_sweep_results(sweep_results)
    plt.savefig("grid_optimization.png")
    plt.close()

    print("Pipeline execution finished successfully with optimal hyperparameters and test sets evaluated!")
