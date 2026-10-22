"""Fast regression tests for data preparation, modelling, and the CLI.

All tests use synthetic data; the full DTMB 5415 download is not required.
"""
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_prep import load_and_preprocess_data
from neural_network import (
    evaluate_pytorch_model,
    regression_metrics,
    train_optimized_mlp,
)


def write_mat(path: Path, X, coefficients):
    """Write arrays using the same orientation as the real MATLAB dataset."""
    scipy.io.savemat(
        path,
        {
            "variablesMatrix": np.asarray(X, dtype=float),
            "coefStatMatrix": np.asarray(coefficients, dtype=float),
        },
    )


@pytest.fixture
def synthetic_mat(tmp_path):
    """Create a small, finite regression dataset with known structure."""
    rng = np.random.default_rng(123)
    X = rng.normal(size=(4, 80))
    coefficients = np.vstack(
        [0.4 * X[0] - 0.2 * X[1] + 0.03 * rng.normal(size=80),
         X[2] + X[3]]
    )
    path = tmp_path / "synthetic.mat"
    write_mat(path, X, coefficients)
    return path, X, coefficients


def test_preprocessing_shapes_and_split_sizes(synthetic_mat):
    path, X, _ = synthetic_mat
    data = load_and_preprocess_data(path, target_row=0)
    assert data["X_train"].shape[1] == X.shape[0]
    assert data["n_samples"] == X.shape[1]
    assert data["n_features"] == X.shape[0]
    assert sum(data["split_sizes"].values()) == X.shape[1]
    assert data["X_train"].dtype == np.float32
    assert np.isfinite(data["X_train"]).all()


@pytest.mark.parametrize("target_row", [-1, 2, 99])
def test_rejects_target_row_out_of_bounds(synthetic_mat, target_row):
    path, _, _ = synthetic_mat
    with pytest.raises(ValueError, match="target_row"):
        load_and_preprocess_data(path, target_row=target_row)


def test_rejects_mismatched_hull_counts(tmp_path):
    path = tmp_path / "bad_shapes.mat"
    write_mat(path, np.ones((3, 10)), np.ones((2, 9)))
    with pytest.raises(ValueError, match="hull counts differ"):
        load_and_preprocess_data(path)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_rejects_non_finite_features(tmp_path, bad_value):
    X = np.ones((3, 12))
    X[1, 4] = bad_value
    path = tmp_path / "nonfinite_x.mat"
    write_mat(path, X, np.ones((2, 12)))
    with pytest.raises(ValueError, match="NaN or infinite"):
        load_and_preprocess_data(path)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_rejects_non_finite_targets(tmp_path, bad_value):
    y = np.ones((2, 12))
    y[0, 4] = bad_value
    path = tmp_path / "nonfinite_y.mat"
    write_mat(path, np.ones((3, 12)), y)
    with pytest.raises(ValueError, match="NaN or infinite"):
        load_and_preprocess_data(path)


def test_scalers_are_fitted_on_training_partition_only(synthetic_mat):
    path, X, coefficients = synthetic_mat
    seed = 17
    test_size, val_size = 0.20, 0.10
    indices = np.arange(X.shape[1])
    train_val_idx, _ = train_test_split(
        indices, test_size=test_size, random_state=seed
    )
    relative_val_size = val_size / (1.0 - test_size)
    train_idx, _ = train_test_split(
        train_val_idx, test_size=relative_val_size, random_state=seed
    )

    data = load_and_preprocess_data(path, target_row=0, random_state=seed)
    np.testing.assert_allclose(data["x_scaler"].mean_, X[:, train_idx].T.mean(axis=0))
    np.testing.assert_allclose(data["y_scaler"].mean_, [coefficients[0, train_idx].mean()])

    # A deliberate extreme value outside the training split must not influence
    # the scaler statistics.
    assert np.allclose(data["x_scaler"].mean_, X[:, train_idx].T.mean(axis=0))


def test_mlp_training_and_prediction_interface(synthetic_mat):
    path, _, _ = synthetic_mat
    data = load_and_preprocess_data(path, target_row=0, random_state=7)
    model, history = train_optimized_mlp(
        data["X_train"],
        data["y_train_scaled"],
        data["X_val"],
        data["y_val_scaled"],
        hidden_sizes=(8,),
        epochs=3,
        batch_size=16,
        patience=2,
        seed=7,
    )
    predictions = evaluate_pytorch_model(model, data["X_test"], data["y_scaler"])
    assert predictions.shape == data["y_test"].shape
    assert np.isfinite(predictions).all()
    assert 1 <= history["best_epoch"] <= len(history["train_loss"]) <= 3
    assert len(history["train_loss"]) == len(history["val_loss"])


def test_regression_metrics_known_values():
    actual = np.array([1.0, 2.0, 3.0])
    predicted = np.array([1.0, 2.0, 2.0])
    metrics = regression_metrics(actual, predicted)
    assert metrics["MSE"] == pytest.approx(1.0 / 3.0)
    assert metrics["RMSE"] == pytest.approx(np.sqrt(1.0 / 3.0))
    assert metrics["MAE"] == pytest.approx(1.0 / 3.0)
    assert metrics["R2"] == pytest.approx(0.5)


def test_cli_smoke_run_without_real_dataset(tmp_path, synthetic_mat):
    path, _, _ = synthetic_mat
    project_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "smoke_outputs"
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    result = subprocess.run(
        [
            sys.executable, str(project_root / "main.py"),
            "--data", str(path),
            "--target-row", "0",
            "--skip-tuning",
            "--epochs", "2",
            "--patience", "1",
            "--output-dir", str(output_dir),
        ],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"CLI failed with exit code {result.returncode}.\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert (output_dir / "metrics.json").is_file()
    assert (output_dir / "training_validation_loss.png").is_file()
    assert (output_dir / "test_predictions.png").is_file()
