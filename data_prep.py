"""Load the DTMB 5415 MATLAB dataset and create leakage-safe data splits.

IMPORTANT: target_row defaults to 0 only as a configurable starting point. Verify
the physical meaning of coefStatMatrix rows against the dataset documentation.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data(
    mat_file_path,
    target_row=0,
    test_size=0.20,
    val_size=0.10,
    random_state=42,
):
    """Load data and return scaled features plus original/scaled targets.

    The test split is isolated before fitting either scaler. The validation
    fraction is expressed as a fraction of the complete dataset.
    """
    path = Path(mat_file_path)
    if not path.is_file():
        raise FileNotFoundError(f"MATLAB dataset not found: {path}")

    mat_data = scipy.io.loadmat(path)
    required = ("variablesMatrix", "coefStatMatrix")
    missing = [key for key in required if key not in mat_data]
    if missing:
        raise KeyError(f"MATLAB file is missing required arrays: {missing}")

    raw_x = np.asarray(mat_data["variablesMatrix"], dtype=np.float64)
    raw_y = np.asarray(mat_data["coefStatMatrix"], dtype=np.float64)

    # Expected database layout: variables x hulls, coefficient rows x hulls.
    if raw_x.ndim != 2 or raw_y.ndim != 2:
        raise ValueError("variablesMatrix and coefStatMatrix must both be 2-D arrays.")
    if raw_x.shape[1] != raw_y.shape[1]:
        raise ValueError(
            "Feature/target hull counts differ: "
            f"{raw_x.shape[1]} vs {raw_y.shape[1]}."
        )
    if not 0 <= target_row < raw_y.shape[0]:
        raise ValueError(
            f"target_row={target_row} is invalid; coefStatMatrix has "
            f"{raw_y.shape[0]} rows (valid indices 0..{raw_y.shape[0] - 1})."
        )
    if not 0 < test_size < 1 or not 0 < val_size < 1 or test_size + val_size >= 1:
        raise ValueError("Require 0 < test_size < 1, 0 < val_size < 1, and their sum < 1.")

    X = raw_x.T
    y = raw_y[target_row, :]

    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError(
            "Dataset contains NaN or infinite values. Investigate these records "
            "rather than silently imputing or dropping them."
        )

    indices = np.arange(len(y))
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state
    )
    relative_val_size = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=relative_val_size, random_state=random_state
    )

    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    # Fit preprocessing on training data only.
    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))

    X_train_scaled = x_scaler.transform(X_train).astype(np.float32)
    X_val_scaled = x_scaler.transform(X_val).astype(np.float32)
    X_test_scaled = x_scaler.transform(X_test).astype(np.float32)

    y_train_scaled = y_scaler.transform(y_train.reshape(-1, 1)).astype(np.float32).ravel()
    y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).astype(np.float32).ravel()

    feature_names = [f"Param_{i + 1}" for i in range(X.shape[1])]
    return {
        "X_train": X_train_scaled,
        "X_val": X_val_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "y_train_scaled": y_train_scaled,
        "y_val_scaled": y_val_scaled,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "feature_names": feature_names,
        "target_row": target_row,
        "n_samples": len(y),
        "n_features": X.shape[1],
        "split_sizes": {
            "train": len(train_idx),
            "validation": len(val_idx),
            "test": len(test_idx),
        },
    }
