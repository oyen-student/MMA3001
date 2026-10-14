
# DTMB 5415 Hull Form Wave Resistance Optimization

This repository provides a modular, production-ready pipeline for predicting the Wave Resistance Coefficient ($C_w$) using machine learning. It explores standard Linear Regression, exhaustive Grid Search parameter sweeps, and Bayesian Optimization via Optuna.

## Structure
- `data_prep.py`: Preprocessing, standard scaling, and train-validation-test split partitioning.
- `linear_regression.py`: Linear regression baseline model training.
- `neural_network.py`: Optimized Deep MLP training, grid-search parameters sweeps, and logging comparison.
- `hyperparameter_tuning.py`: Smart Bayesian Hyperparameter optimization with Optuna.
- `main.py`: Unified pipeline execution script.
