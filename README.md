# DTMB 5415 Hull-Resistance Prediction (MMA3001)

This repository compares simple regression baselines with a PyTorch multilayer perceptron (MLP) for predicting a selected row of `coefStatMatrix` from the DTMB 5415 hull parameter matrix.

> **Important scientific check:** `data_prep.py` defaults to `target_row=0` for compatibility with the original draft. Do not interpret this as wave-resistance coefficient (C_w) until the DTMB 5415 dataset documentation confirms what each row of `coefStatMatrix` means. Similarly, input columns are named `Param_1`, etc. until their physical definitions and units are verified. The code deliberately records this unresolved check in `outputs/metrics.json`.

## Pipeline

- `data_prep.py`: validates MAT-file keys/shapes and finite values; creates reproducible train/validation/test splits; fits feature and target standardisers on training data only.
- `linear_regression.py`: mean-target sanity baseline and linear-regression baseline.
- `neural_network.py`: PyTorch MLP, reproducible training, validation-loss tracking, early stopping, best-checkpoint restoration, regression metrics, and diagnostic plots.
- `hyperparameter_tuning.py`: Optuna TPE search over MLP architecture, learning rate, weight decay, and batch size using validation MSE in original target units.
- `main.py`: complete experiment, held-out test evaluation, and output saving.
- `requirements.txt`: Python dependencies.

## Setup

Use a virtual environment if possible, then install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run

Run the full pipeline with Optuna tuning:

```bash
python main.py --data "/path/to/DTMB5415-database.mat" --target-row 0 --trials 20
```

Add `--sweep` to run an additional architecture comparison and save `architecture_sweep.csv` and `architecture_sweep.png`. This performs extra training and is optional.\n\nFor example, run tuning and the architecture sweep together:\n\n```bash\npython main.py --data "/path/to/DTMB5415-database.mat" --trials 20 --sweep\n```\n\nRun a faster initial check without Optuna:

```bash
python main.py --data "/path/to/DTMB5415-database.mat" --target-row 0 --skip-tuning
```

The path is supplied on the command line instead of being hard-coded to a particular Google Drive location. On Windows, quote the full path if it contains spaces.

## Outputs

The output directory defaults to `outputs/` and contains:

- `metrics.json`: validation and held-out test MSE, RMSE, MAE and (R^2), sample counts, selected configuration, and a flag noting that target-row semantics still require verification.
- `training_validation_loss.png`: training and validation loss with best epoch marked.\n- `architecture_sweep.csv` and `architecture_sweep.png`: actual training/validation errors for the optional architecture sweep.
- `test_predictions.png`: held-out actual-versus-predicted plot.
- `test_residuals.png`: held-out residual plot.
- `optuna_convergence.png`: validation-objective search history (when tuning is enabled).

## Methodological safeguards

1. **No preprocessing leakage:** feature and target scalers are fitted on the training split only.
2. **Validation for model selection:** validation data is used for early stopping and Optuna selection; it is not used as the final performance estimate.
3. **Held-out test:** the test split is evaluated after model selection. Do not use test results to choose the architecture or tune hyperparameters.
4. **Consistent metrics:** model comparisons use MSE, RMSE, MAE and (R^2), all calculated in the target's original units.
5. **Reproducibility:** the train/validation/test split and PyTorch/Optuna sampling use fixed seeds by default.
6. **Interpretation limits:** random splitting assumes samples can be treated as independent draws. If hulls are related variants or grouped designs, consider a group-aware split to test generalisation to genuinely unseen hull families.

## Before writing conclusions

- Verify which `coefStatMatrix` row is the intended physical target, its units, and whether it truly represents (C_w).
- Find the definitions and units for each input variable. The generic names are not physical explanations.
- Inspect the split and data distribution, and check for duplicate/near-duplicate hulls across splits.
- Compare the mean baseline, linear model, and MLP test results. A neural network is not automatically better.
- Do not claim Optuna found a global optimum; it finds the best configuration among the trials performed.
- Run the code against the actual MAT file before reporting results. The repository update is a code revision, not a reproduced numerical experiment.
