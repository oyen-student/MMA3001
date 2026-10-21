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

## Dataset download and setup

The MATLAB dataset is approximately 1 GB, so it is **not stored in this GitHub repository**. Download it separately from Zenodo:

**Dataset record:** https://zenodo.org/records/19222008

1. Open the Zenodo record above and download the dataset archive.
2. Extract/unzip the downloaded archive. The extracted folder should be named `dr-benchmark-5415hull`.
3. Move the **entire extracted `dr-benchmark-5415hull` folder** into the `data` folder in this repository. Do not move only the MAT file out of that folder.
4. Confirm that the MATLAB file is located at this exact path:

   `data/dr-benchmark-5415hull/DTMB5415-database.mat`

After setup, the relevant part of the project directory should look like this:

```text
MMA3001/
├── data/
│   └── dr-benchmark-5415hull/
│       └── DTMB5415-database.mat
├── main.py
├── data_prep.py
├── neural_network.py
├── linear_regression.py
├── hyperparameter_tuning.py
├── requirements.txt
└── README.md
```

The downloaded dataset is excluded from Git tracking because of its size. The `data` folder in GitHub contains only a placeholder; you must download and extract the dataset yourself before running the pipeline.

## Install dependencies

Use a virtual environment if possible, then install dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

## Run the pipeline

Once the dataset is in the location above, the program finds it automatically. Run the pipeline with Optuna tuning:

```bash
python main.py --target-row 0 --trials 20
```

Add `--sweep` to run an additional architecture comparison and save `architecture_sweep.csv` and `architecture_sweep.png`. This performs extra training and is optional:

```bash
python main.py --trials 20 --sweep
```

For a faster initial check without Optuna:

```bash
python main.py --skip-tuning
```

### Using a different dataset location

If you have stored the extracted folder somewhere else, pass the full path to the MAT file with `--data`:

```bash
python main.py --data "/path/to/DTMB5415-database.mat" --skip-tuning
```

On Windows, quote the full path if it contains spaces.

### Running in Google Colab

You can leave the dataset in Google Drive instead of copying it into the Colab repository directory. Mount Drive, clone the repository, install the dependencies, and pass the full Drive path to the MAT file:

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
!git clone https://github.com/oyen-student/MMA3001.git
%cd MMA3001
!pip install -r requirements.txt
!python main.py --data "/content/drive/MyDrive/path/to/DTMB5415-database.mat" --skip-tuning
```

Replace the example path with the actual location of `DTMB5415-database.mat` in your mounted Drive.

## Outputs

The output directory defaults to `outputs/` and contains:

- `metrics.json`: validation and held-out test MSE, RMSE, MAE and (R²), sample counts, selected configuration, and a flag noting that target-row semantics still require verification.
- `training_validation_loss.png`: training and validation loss with best epoch marked.
- `architecture_sweep.csv` and `architecture_sweep.png`: actual training/validation errors for the optional architecture sweep.
- `test_predictions.png`: held-out actual-versus-predicted plot.
- `test_residuals.png`: held-out residual plot.
- `optuna_convergence.png`: validation-objective search history (when tuning is enabled).

## Methodological safeguards

1. **No preprocessing leakage:** feature and target scalers are fitted on the training split only.
2. **Validation for model selection:** validation data is used for early stopping and Optuna selection; it is not used as the final performance estimate.
3. **Held-out test:** the test split is evaluated after model selection. Do not use test results to choose the architecture or tune hyperparameters.
4. **Consistent metrics:** model comparisons use MSE, RMSE, MAE and (R²), all calculated in the target's original units.
5. **Reproducibility:** the train/validation/test split and PyTorch/Optuna sampling use fixed seeds by default.
6. **Interpretation limits:** random splitting assumes samples can be treated as independent draws. If hulls are related variants or grouped designs, consider a group-aware split to test generalisation to genuinely unseen hull families.

## Before writing conclusions

- Verify which `coefStatMatrix` row is the intended physical target, its units, and whether it truly represents (C_w).
- Find the definitions and units for each input variable. The generic names are not physical explanations.
- Inspect the split and data distribution, and check for duplicate/near-duplicate hulls across splits.
- Compare the mean baseline, linear model, and MLP test results. A neural network is not automatically better.
- Do not claim Optuna found a global optimum; it finds the best configuration among the trials performed.
- Run the code against the actual MAT file before reporting results. The repository update is a code revision, not a reproduced numerical experiment.
