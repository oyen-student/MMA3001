"""Load the DTMB 5415 MATLAB dataset and create leakage-safe data splits.

IMPORTANT: target_row defaults to 0 only as a configurable starting point. Verify
the physical meaning of coefStatMatrix rows against the dataset documentation.
"""
from pathlib import Path

import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# The archive's metadata.txt numbers rows from one. Python's target_row is
# zero-based, so row 0 below corresponds to metadata row 1.
TARGET_ROW_DEFINITIONS = {
    0: {
        "quantity": "Wave resistance coefficient (Cw)",
        "froude_number": 0.25,
        "scale": "model scale",
        "source": "DTMB5415 dataset metadata.txt (included with the dataset archive)",
    },
    1: {
        "quantity": "Wave resistance coefficient (Cw)",
        "froude_number": 0.33,
        "scale": "model scale",
        "source": "DTMB5415 dataset metadata.txt (included with the dataset archive)",
    },
    2: {
        "quantity": "Wave resistance coefficient (Cw)",
        "froude_number": 0.41,
        "scale": "model scale",
        "source": "DTMB5415 dataset metadata.txt (included with the dataset archive)",
    },
}


def load_and_preprocess_data(
    mat_file_path,
    target_row=0,
    test_size=0.20,
    val_size=0.10,
    random_state=42,
    exclude_training_sample_indices=(),
):
    """Load the MATLAB database and prepare leakage-safe data partitions.

    Features are transposed so each row represents one hull. The test split is
    separated before the scalers are fitted; validation size is expressed as a
    fraction of the complete dataset.

    Args:
        mat_file_path (str or Path): Path to DTMB5415-database.mat.
        target_row (int): Row index from coefStatMatrix to predict. Verify its
            physical definition in the dataset documentation before interpreting
            results.
        test_size (float): Fraction of all samples reserved for the held-out test set.
        val_size (float): Fraction of all samples reserved for validation.
        random_state (int): Seed controlling reproducible random splitting.
        exclude_training_sample_indices (iterable[int]): Optional zero-based
            source sample indices to omit from training after the split is made.
            Validation and test membership remain unchanged.

    Returns:
        dict: Scaled feature arrays, original and scaled target arrays, fitted
        scalers, feature labels, target-row metadata, and split sizes. Scaled
        test features are returned for consistent preprocessing; scaled test
        targets are intentionally not used for model selection.

    Raises:
        FileNotFoundError: If the requested MAT file does not exist.
        KeyError: If required arrays are absent from the MAT file.
        ValueError: If array dimensions, sample counts, split sizes, or values
            are invalid.
    """
    path = Path(mat_file_path)
    if not path.is_file():
        raise FileNotFoundError(f"MATLAB dataset not found: {path}")

    # Load only the requested MATLAB archive, then check required keys before
    # indexing so an unexpected archive fails with a useful message.
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

    # MATLAB stores one hull per column; scikit-learn expects one hull per row.
    X = raw_x.T
    y = raw_y[target_row, :]

    # Fail instead of imputing: silent repairs would alter the dataset and
    # make the reported split and model results harder to reproduce.
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError(
            "Dataset contains NaN or infinite values. Investigate these records "
            "rather than silently imputing or dropping them."
        )

    # Split integer sample IDs first so the exact memberships can be audited
    # and reused when a named training observation is excluded below.
    indices = np.arange(len(y))
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state
    )
    # val_size is defined against the full dataset, but this second split sees
    # only the post-test remainder; convert it to that remainder's fraction.
    relative_val_size = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=relative_val_size, random_state=random_state
    )

    # Apply exclusions after splitting so train/validation/test membership is
    # unchanged between paired sensitivity runs.
    excluded_indices = [int(index) for index in exclude_training_sample_indices]
    if len(set(excluded_indices)) != len(excluded_indices):
        raise ValueError("Excluded training sample indices must be unique.")
    if any(index < 0 or index >= len(y) for index in excluded_indices):
        raise ValueError("An excluded training sample index is outside the dataset.")
    not_in_training = sorted(set(excluded_indices) - set(train_idx.tolist()))
    if not_in_training:
        raise ValueError(
            "Only samples assigned to the training split can be excluded; "
            f"these indices are not in training: {not_in_training}"
        )
    if len(excluded_indices) >= len(train_idx):
        raise ValueError("At least one training sample must remain after exclusions.")
    excluded_set = set(excluded_indices)
    train_idx = np.asarray(
        [index for index in train_idx if int(index) not in excluded_set], dtype=int
    )

    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    # Audit exact duplicate design vectors without using target values. This
    # reports whether random splitting placed an identical design in multiple
    # partitions; it does not define a near-duplicate tolerance or group hulls.
    split_by_sample = np.empty(len(y), dtype=object)
    split_by_sample[train_idx] = "train"
    split_by_sample[val_idx] = "validation"
    split_by_sample[test_idx] = "test"
    if excluded_indices:
        split_by_sample[np.asarray(excluded_indices, dtype=int)] = "excluded"
    _, duplicate_inverse, duplicate_counts = np.unique(
        X, axis=0, return_inverse=True, return_counts=True
    )
    duplicate_group_indices = np.flatnonzero(duplicate_counts > 1)
    cross_split_duplicate_groups = 0
    cross_split_duplicate_samples = 0
    # duplicate_inverse maps each row to its exact-equality group. Excluded
    # rows do not count as leakage because they never reach a fitted model.
    for group_index in duplicate_group_indices:
        members = np.flatnonzero(duplicate_inverse == group_index)
        active_members = members[split_by_sample[members] != "excluded"]
        if np.unique(split_by_sample[active_members]).size > 1:
            cross_split_duplicate_groups += 1
            cross_split_duplicate_samples += len(active_members)
    duplicate_audit = {
        "method": "exact equality of normalized design-variable rows",
        "duplicate_groups": int(len(duplicate_group_indices)),
        "samples_in_duplicate_groups": int(duplicate_counts[duplicate_counts > 1].sum()),
        "duplicate_groups_crossing_splits": int(cross_split_duplicate_groups),
        "samples_in_cross_split_duplicate_groups": int(cross_split_duplicate_samples),
        "near_duplicates_audited": False,
    }

    # Fit preprocessing on training data only. Validation and test samples
    # are transformed with these statistics, avoiding information leakage.
    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))

    X_train_scaled = x_scaler.transform(X_train).astype(np.float32)
    X_val_scaled = x_scaler.transform(X_val).astype(np.float32)
    X_test_scaled = x_scaler.transform(X_test).astype(np.float32)

    y_train_scaled = y_scaler.transform(y_train.reshape(-1, 1)).astype(np.float32).ravel()
    y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).astype(np.float32).ravel()
    # Test targets remain in original units: they never affect fitting, early
    # stopping, or hyperparameter selection.

    # The archive identifies normalized design parameters by position rather
    # than giving a verified physical name for each one.
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
        "target_definition": TARGET_ROW_DEFINITIONS.get(target_row),
        "excluded_training_sample_indices": excluded_indices,
        "n_samples": len(y),
        "n_features": X.shape[1],
        "split_sizes": {
            "train": len(train_idx),
            "validation": len(val_idx),
            "test": len(test_idx),
        },
        "duplicate_audit": duplicate_audit,
    }
