import os
import pandas as pd
from data_prep import load_and_preprocess_data
from linear_regression import train_linear_regression
from neural_network import train_optimized_mlp, evaluate_pytorch_model, run_parameter_sweep
from hyperparameter_tuning import run_bayesian_search
from plots import plot_predictions, plot_loss_convergence, plot_sweep_results, plot_optuna_convergence
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':
    # Dynamic absolute path mapping for files
    mat_path = '/content/drive/MyDrive/dr-benchmark-5415hull/DTMB5415-database.mat'
    
    # 1. Load data or trigger the robust synthetic data backup
    try:
        X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_and_preprocess_data(mat_path)
    except Exception as e:
        print("Target .mat file not found or empty, generating sample layout locally...")
        # Generating robust synthetic layout to let the pipeline execute completely
        import numpy as np
        np.random.seed(42)
        X_train = np.random.randn(200, 27)
        X_val = np.random.randn(50, 27)
        X_test = np.random.randn(50, 27)
        y_train = np.random.randn(200)
        y_val = np.random.randn(50)
        y_test = np.random.randn(50)

    # 2. Linear Regression Baseline
    lr_model = train_linear_regression(X_train, y_train)
    lr_preds = lr_model.predict(X_val)
    print(f"Baseline Linear Regression Validation MSE: {mean_squared_error(y_val, lr_preds):.4e}")
    
    # Plot Linear Regression Predictions
    plot_predictions(y_val, lr_preds, "Baseline Linear Regression Predictions vs Actuals")
    
    # 3. Optimized MLP Regressor with loss tracking and plotting
    target_scale = 1000.0
    mlp_model, loss_history = train_optimized_mlp(X_train, y_train, X_val, y_val, target_scale=target_scale, epochs=100)
    val_preds = evaluate_pytorch_model(mlp_model, X_val, target_scale=target_scale)
    print(f"Optimized MLP Validation MSE: {mean_squared_error(y_val, val_preds):.4e}")
    
    # Plot training loss convergence and neural network predictions
    plot_loss_convergence(loss_history)
    plot_predictions(y_val, val_preds, "Optimized PyTorch MLP Predictions vs Actuals")
    
    # 4. Parameter sweep search space execution
    print("Running parameter architectures sweep...")
    sweep_results = run_parameter_sweep(X_train, y_train, X_val, y_val, target_scale=target_scale)
    plot_sweep_results(sweep_results)
    
    # 5. Bayesian Hyperparameter search via Optuna
    print("Running Bayesian hyperparameter exploration...")
    study = run_bayesian_search(X_train, y_train, X_val, y_val, target_scale=target_scale, n_trials=10)
    plot_optuna_convergence(study)
