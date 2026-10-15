
import os
from data_prep import load_and_preprocess_data
from linear_regression import train_linear_regression
from neural_network import train_optimized_mlp, plot_predictions, run_parameter_sweep, plot_sweep_results
from hyperparameter_tuning import run_bayesian_search, plot_optuna_convergence
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':
    mat_path = '/content/drive/MyDrive/dr-benchmark-5415hull/DTMB5415-database.mat'
    if not os.path.exists(mat_path):
         print("Please update mat_file_path to point to your DTMB5415-database.mat file.")
         exit()
         
    # 1. Load and prep data
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_and_preprocess_data(mat_path)
    
    # 2. Linear Regression Baseline
    lr_model = train_linear_regression(X_train, y_train)
    lr_preds = lr_model.predict(X_val)
    print(f"Baseline Linear Regression Validation MSE: {mean_squared_error(y_val, lr_preds):.4e}")
    
    # 3. Optimized MLP Regressor
    target_scale = 1000.0
    mlp_model = train_optimized_mlp(X_train, y_train, X_val, y_val, target_scale=target_scale)
    val_preds_scaled = mlp_model.predict(X_val)
    val_preds = val_preds_scaled / target_scale
    print(f"Optimized MLP Validation MSE: {mean_squared_error(y_val, val_preds):.4e}")
