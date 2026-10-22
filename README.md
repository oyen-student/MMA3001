# DTMB 5415 Hull-Resistance Prediction (MMA3001)

This repository compares simple regression baselines with a PyTorch multilayer perceptron (MLP) for predicting a selected row of `coefStatMatrix` from the DTMB 5415 hull parameter matrix.

`data_prep.py` uses zero-based Python row indices. The `metadata.txt` included in the dataset archive maps `coefStatMatrix` rows 1-3 to model-scale wave-resistance coefficient (`Cw`) at Froude numbers 0.25, 0.33, and 0.41. Thus `--target-row 0` means `Cw` at Fr=0.25, according to that metadata. The result JSON records this definition and its source. The 27 inputs remain normalized design variables; individual variable meanings and units are not established by the metadata inspected here.

## Pipeline

- `data_prep.py`: validates MAT-file keys/shapes and finite values; creates reproducible random sample-level train/validation/test splits; fits feature and target standardisers on training data only; records the target-row definition; and audits exact duplicate design vectors across splits.
- `linear_regression.py`: mean-target sanity baseline and linear-regression baseline.
- `neural_network.py`: PyTorch MLP, reproducible training, post-epoch train/validation loss measured in evaluation mode, configurable gradient clipping, pre-clipping gradient-norm diagnostics, early stopping, best-checkpoint restoration, regression metrics, and diagnostic plots.
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

The downloaded dataset is excluded from Git tracking because of its size. Download and extract it before running the pipeline. The repository also contains older `data/train.csv`, `data/val.csv`, and `data/test.csv` files from the earlier preprocessing workflow. Their provenance is not recorded, and the current CLI does not read them; do not use them as model inputs or evidence of the current experiment. The `src/data_prep.py` compatibility wrapper now delegates to the canonical loader and cannot synthesize fallback data.

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

Training uses a maximum of 500 epochs per model and stops early if validation loss does not improve by at least `--min-delta` for 50 consecutive epochs. Gradient clipping defaults to a global norm of 1.0; set `--grad-clip 0` to disable clipping. These are experimental stability controls, not a guarantee that training will be stable. Adjust them with `--epochs`, `--patience`, `--min-delta`, and `--grad-clip`; for example:

```bash
python main.py --target-row 0 --trials 20 --epochs 1000 --patience 100 --min-delta 1e-6 --grad-clip 1.0
```

### Architecture sweep and Optuna tuning

These are two distinct kinds of model-selection work. In the CLI, `--sweep` runs an architecture comparison on the **training and validation partitions**. It trains ten preset hidden-layer layouts using the same default learning rate, weight decay, and batch size, ranks them by validation MSE, and writes `architecture_sweep.csv` and `architecture_sweep.png`. It prints the best sweep architecture, but does not pass that architecture into the final model configuration.

Optuna (enabled by default) searches a broader MLP configuration: one to three hidden layers, layer widths, learning rate, weight decay, and batch size. `--trials` controls the number of configurations trained. The best Optuna configuration is then trained again and used for final evaluation. `optuna_convergence.png` shows each trial's validation objective and the best score so far; it documents the search, but does not establish that the search found a global optimum or fully converged.

The sweep and Optuna both train MLPs and both search architecture choices. The sweep tests ten preset architectures while holding optimizer settings fixed. Optuna searches architecture and optimizer settings together. Running both is supported, but adds the sweep's ten training runs before Optuna and its final refit. The sweep result remains a diagnostic and does not change the Optuna-selected or fixed final model. For normal tuning, Optuna alone is usually sufficient.

To run the architecture sweep without Optuna, add both `--sweep` and `--skip-tuning`:

```bash
python main.py --target-row 0 --sweep --skip-tuning
```

To run Optuna and the architecture sweep in the same experiment:

```bash
python main.py --target-row 0 --trials 20 --sweep
```

To run only the fixed default MLP and skip Optuna, use:

```bash
python main.py --target-row 0 --skip-tuning
```

**Neither the architecture sweep nor Optuna uses the held-out test partition to choose configurations.** Both use validation results. The held-out test set is evaluated afterward for final reporting; do not use its metrics to pick between sweep, Optuna, or fixed-model results.

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

## Automated tests

The test suite uses a small synthetic MATLAB-format dataset, so it does **not** require downloading the approximately 1 GB DTMB dataset or running Optuna tuning.

Install runtime and test dependencies:

```bash
python -m pip install -r requirements-dev.txt
```

From the repository root, parse project Python files without writing bytecode, then run pytest:

```bash
python -B check_syntax.py
python -m pytest -q
```

The tests cover target-row bounds and metadata, mismatched MATLAB array dimensions, NaN/infinity rejection, train-only scaler fitting with an extreme held-out outlier, exact duplicate detection across splits, Optuna trial validation, the MLP training/prediction interface, known-value regression metrics, the deprecated loader's missing-file behavior, CLI input validation, and a short end-to-end CLI smoke test. GitHub Actions is configured to run these checks on pushes and pull requests. A passing synthetic-data test suite checks core software behavior; it does not validate the real MAT file or independently establish the correctness of its metadata.

## Outputs

The output directory defaults to `outputs/` and contains:

- `metrics.json`: validation and held-out test MSE, RMSE, MAE and R-squared; executed command and Python version; dataset/output paths; target-row definition and source; split fractions and sample counts; seed; training/tuning settings; selected configuration; and exact-duplicate audit counts.
- `training_validation_loss.png`: training and validation loss measured after each epoch's updates, with best epoch marked.
- `gradient_norms.png`: mean and maximum minibatch gradient norms before clipping, to help diagnose unstable optimisation.
- `architecture_sweep.csv` and `architecture_sweep.png`: actual training/validation errors for the optional architecture sweep.
- `test_predictions.png`: held-out actual-versus-predicted plot.
- `test_model_comparison.png`: linear-regression and MLP predictions on the same held-out actual-versus-predicted plot, with R-squared and RMSE in the legend.
- `test_residuals.png`: held-out residual plot.
- `optuna_convergence.png`: validation-objective search history (when tuning is enabled).

## Methodological safeguards

1. **No preprocessing leakage:** feature and target scalers are fitted on the training split only.
2. **Validation for model selection:** validation data is used for early stopping and Optuna selection; it is not used as the final performance estimate.
3. **Held-out test:** the test split is evaluated after model selection. Do not use test results to choose the architecture or tune hyperparameters.
4. **Comparable loss curves:** training and validation losses are both measured after epoch updates with the model in evaluation mode.
5. **Gradient diagnostics:** gradient norms are recorded before clipping; clipping can reduce extreme updates but does not diagnose or eliminate data issues.
6. **Consistent metrics:** model comparisons use MSE, RMSE, MAE and (R²), all calculated in the target's original units.
7. **Reproducibility:** the train/validation/test split and PyTorch/Optuna sampling use fixed seeds by default.
8. **Interpretation limits:** this implementation uses a random sample-level split and reports exact duplicate design vectors crossing partitions. Near duplicates are not measured because there is no justified tolerance, and no hull-family labels are available for a group-aware split. The score estimates performance on this split, not necessarily on genuinely unseen hull families.

## Before writing conclusions

- Cite the included dataset `metadata.txt` for row identity and scale; verify against the dataset publication if independent confirmation is required.
- Find the definitions and units for each input variable. The generic names are not physical explanations.
- Inspect exact-duplicate audit counts in `metrics.json`; choose and justify a near-duplicate tolerance and obtain hull-family labels before claiming generalization to new hull families.
- Compare the mean baseline, linear model, and MLP test results. A neural network is not automatically better.
- Do not claim Optuna found a global optimum; it finds the best configuration among the trials performed.
- Treat checked-in output folders as saved experiment artifacts, not freshly reproduced results. A new reportable run should record its command, seed, target row, split sizes, tuning settings, environment, metrics, and output directory.
