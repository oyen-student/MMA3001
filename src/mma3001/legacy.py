"""Compatibility wrapper for the canonical package data loader.

The old implementation could silently substitute random values when the MAT
file was missing. This module now delegates to :mod:`data_prep`, which raises
an error for missing or invalid input and uses the same split/scaling logic as
the main CLI. New code should import from :mod:`mma3001.data_prep`.
"""
from pathlib import Path
import warnings

import pandas as pd

from .data_prep import load_and_preprocess_data as _load_and_preprocess_data


def load_and_preprocess_data(mat_file_path, output_dir=None, **kwargs):
    """Load real data through the canonical loader, optionally exporting CSVs.

    Args:
        mat_file_path: Existing MATLAB dataset path.
        output_dir: Optional directory for scaled train/validation/test CSVs.
        **kwargs: Keyword arguments accepted by the canonical loader.

    Returns:
        The legacy six-item tuple of scaled feature splits and original-unit
        target splits. No synthetic or fallback dataset is ever generated.
    """
    warnings.warn(
        "mma3001.legacy is deprecated; import load_and_preprocess_data from "
        "mma3001.data_prep instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    data = _load_and_preprocess_data(mat_file_path, **kwargs)

    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        for split in ("train", "validation", "test"):
            x = data[f"X_{'val' if split == 'validation' else split}"]
            y = data[f"y_{'val' if split == 'validation' else split}"]
            frame = pd.DataFrame(x)
            frame["target"] = y
            filename = "val.csv" if split == "validation" else f"{split}.csv"
            frame.to_csv(destination / filename, index=False)

    return (
        data["X_train"], data["X_val"], data["X_test"],
        data["y_train"], data["y_val"], data["y_test"],
    )
