"""PyTorch MLP regression, early stopping, and evaluation plots."""
from copy import deepcopy
import random
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def set_seed(seed=42):
    """Seed Python, NumPy, and PyTorch random-number generators.

    Args:
        seed: Integer seed used for reproducible initialisation and sampling.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Return the preferred PyTorch device, using CUDA when available.

    Returns:
        A CUDA device when available; otherwise, a CPU device.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PyTorchMLP(nn.Module):
    """Fully connected regression network with ReLU hidden-layer activations."""

    def __init__(self, input_dim, hidden_sizes=(64, 32)):
        """Construct the network from an input width and hidden-layer widths.

        Args:
            input_dim: Number of input features per sample.
            hidden_sizes: Width of each hidden layer, in order.
        """
        super().__init__()
        layers = []
        in_dim = input_dim
        for width in hidden_sizes:
            layers.extend([nn.Linear(in_dim, int(width)), nn.ReLU()])
            in_dim = int(width)
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Run a batch through the network.

        Args:
            x: Tensor shaped (batch_size, input_dim).

        Returns:
            Tensor shaped (batch_size, 1) containing one output per sample.
        """
        return self.network(x)


def train_optimized_mlp(
    X_train,
    y_train_scaled,
    X_val,
    y_val_scaled,
    hidden_sizes=(64, 32),
    epochs=200,
    batch_size=64,
    lr=1e-3,
    weight_decay=1e-4,
    patience=20,
    min_delta=1e-6,
    grad_clip=1.0,
    seed=42,
):
    """Train an MLP with gradient clipping and validation-based early stopping.

    Training and validation losses are measured after each epoch's updates in
    evaluation mode. Gradient norms are recorded before clipping, and the
    weights from the epoch with the lowest qualifying validation loss are
    restored before returning.

    Args:
        X_train (array-like): Scaled training feature matrix.
        y_train_scaled (array-like): Scaled training targets.
        X_val (array-like): Validation features scaled with training statistics.
        y_val_scaled (array-like): Validation targets scaled with training statistics.
        hidden_sizes (tuple[int, ...]): Width of each hidden layer.
        epochs (int): Maximum number of training epochs.
        batch_size (int): Number of samples per minibatch.
        lr (float): Adam optimiser learning rate.
        weight_decay (float): L2-style weight decay passed to Adam.
        patience (int): Epochs without qualifying validation improvement before stopping.
        min_delta (float): Minimum decrease in validation MSE required to reset patience.
        grad_clip (float or None): Maximum global gradient norm; None disables clipping.
        seed (int): Random seed for reproducible initialisation and minibatch order.

    Returns:
        tuple: The restored best-validation PyTorch model and a history dictionary
        containing loss curves, gradient norms, best epoch, and best validation loss.
    """
    set_seed(seed)
    device = get_device()
    X_train_t = torch.as_tensor(np.asarray(X_train), dtype=torch.float32)
    y_train_t = torch.as_tensor(np.asarray(y_train_scaled), dtype=torch.float32).reshape(-1, 1)
    X_val_t = torch.as_tensor(np.asarray(X_val), dtype=torch.float32, device=device)
    y_val_t = torch.as_tensor(
        np.asarray(y_val_scaled).reshape(-1, 1),
        dtype=torch.float32,
        device=device,
    )

    loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=min(batch_size, len(X_train_t)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    model = PyTorchMLP(X_train.shape[1], hidden_sizes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    history = {
        "train_loss": [], "val_loss": [],
        "grad_norm_mean": [], "grad_norm_max": [],
    }
    best_val = float("inf")
    best_state = deepcopy(model.state_dict())
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        epoch_grad_norms = []
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            # Record the global gradient norm before optional clipping.
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip if grad_clip is not None else float("inf"),
            )
            epoch_grad_norms.append(float(grad_norm.detach().cpu()))
            optimizer.step()

        # Measure both losses after the epoch's updates in evaluation mode.
        model.eval()
        with torch.no_grad():
            train_loss = criterion(model(X_train_t.to(device)), y_train_t.to(device)).item()
            val_loss = criterion(model(X_val_t), y_val_t).item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["grad_norm_mean"].append(float(np.mean(epoch_grad_norms)))
        history["grad_norm_max"].append(float(np.max(epoch_grad_norms)))

        if val_loss < best_val - min_delta:
            best_val = val_loss
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    history["best_epoch"] = best_epoch
    history["stopped_epoch"] = len(history["train_loss"])
    history["best_val_loss_scaled"] = best_val
    return model, history


def evaluate_pytorch_model(model, X_data, y_scaler):
    """Predict target values in the original target units.

    Args:
        model (torch.nn.Module): Trained PyTorch model.
        X_data (array-like): Feature matrix transformed with training statistics.
        y_scaler (StandardScaler): Fitted scaler used to reverse target scaling.

    Returns:
        numpy.ndarray: One-dimensional array of predictions in original units.
    """
    device = next(model.parameters()).device
    X_t = torch.as_tensor(np.asarray(X_data), dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        pred_scaled = model(X_t).cpu().numpy().reshape(-1, 1)
    return y_scaler.inverse_transform(pred_scaled).ravel()


def regression_metrics(y_true, y_pred):
    """Calculate standard regression metrics in the supplied target units.

    Args:
        y_true: Ground-truth target values.
        y_pred: Predictions corresponding to y_true.

    Returns:
        Dictionary containing MSE, RMSE, MAE, and coefficient of determination.
    """
    return {
        "MSE": float(mean_squared_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def plot_training_history(history, save_path=None, title_suffix=None):
    """Plot training loss by epoch and distinguish stopping from checkpointing.

    Args:
        history: History dictionary returned by train_optimized_mlp.
        save_path: Optional path at which to save the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], label="Training loss", color="#1f77b4")
    ax.axvline(
        history["stopped_epoch"], linestyle="--", alpha=0.7, color="#555555",
        label=f"Training stopped: epoch {history['stopped_epoch']}",
    )
    ax.text(
        0.02, 0.02,
        f"Final weights restored from epoch {history['best_epoch']}\n"
        "(lowest validation loss; validation curve omitted)",
        transform=ax.transAxes, va="bottom", fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (standardised target)")
    title = "MLP training loss by epoch"
    if title_suffix:
        title = f"{title}\n{title_suffix}"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=160)
    plt.show()


def plot_gradient_norms(history, save_path=None, title_suffix=None):
    """Plot mean and maximum pre-clipping gradient norms per epoch.

    Args:
        history: Training-history dictionary containing gradient-norm series.
        save_path: Optional path at which to save the figure.
    """
    if not history.get("grad_norm_mean"):
        return
    epochs = np.arange(1, len(history["grad_norm_mean"]) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["grad_norm_mean"], label="Mean gradient norm")
    ax.plot(epochs, history["grad_norm_max"], label="Maximum gradient norm", alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Global gradient norm (before clipping)")
    title = "MLP gradient-norm diagnostics"
    if title_suffix:
        title = f"{title}\n{title_suffix}"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=160)
    plt.show()


def plot_predictions(y_true, y_pred, title="Predicted vs actual", save_path=None):
    """Plot predicted targets against actual targets with a 1:1 reference.

    Args:
        y_true: Actual target values in original units.
        y_pred: Predicted target values in original units.
        title: Figure title.
        save_path: Optional path at which to save the figure.
    """
    metrics = regression_metrics(y_true, y_pred)
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.55, s=18)
    ax.plot(
        [lo, hi], [lo, hi], "k--", linewidth=1.5,
        label="1:1 reference (actual = predicted)",
    )
    ax.legend()
    ax.set_xlabel("Actual target")
    ax.set_ylabel("Predicted target")
    ax.set_title(title)
    ax.text(
        0.04, 0.96,
        f"RMSE: {metrics['RMSE']:.4g}\nMAE: {metrics['MAE']:.4g}\nR²: {metrics['R2']:.4f}",
        transform=ax.transAxes, va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=160)
    plt.show()


def plot_train_validation_predictions(
    y_train,
    train_predictions,
    y_val,
    val_predictions,
    save_path=None,
    model_config=None,
    configuration_source=None,
    comparison_label=None,
):
    """Compare final-refit MLP predictions on training and validation data.

    Training points are in-sample; validation points were not used for weight
    updates but may have informed early stopping and hyperparameter selection.
    The figure has a full-range panel with every observation and a zoomed panel
    that makes dense regions visible without dropping points from the full view.

    Args:
        y_train: Training targets in original units.
        train_predictions: MLP predictions for training samples in original units.
        y_val: Validation targets in original units.
        val_predictions: MLP predictions for validation samples in original units.
        save_path: Optional path at which to save the figure.
        model_config: Selected hidden sizes and optimizer settings used by the refit.
        configuration_source: How the configuration was selected or loaded.
        comparison_label: Dataset condition used in this run.
    """
    y_train = np.asarray(y_train)
    train_predictions = np.asarray(train_predictions)
    y_val = np.asarray(y_val)
    val_predictions = np.asarray(val_predictions)

    train_metrics = regression_metrics(y_train, train_predictions)
    val_metrics = regression_metrics(y_val, val_predictions)
    all_values = np.concatenate(
        [y_train, train_predictions, y_val, val_predictions]
    )
    lo = float(np.min(all_values))
    hi = float(np.max(all_values))
    if lo == hi:
        padding = max(abs(lo) * 0.05, 1e-9)
        lo -= padding
        hi += padding

    zoom_lo, zoom_hi = np.percentile(all_values, [0.5, 99.5])
    if zoom_hi <= zoom_lo:
        zoom_lo, zoom_hi = lo, hi

    train_in_zoom = (
        (y_train >= zoom_lo) & (y_train <= zoom_hi)
        & (train_predictions >= zoom_lo) & (train_predictions <= zoom_hi)
    )
    val_in_zoom = (
        (y_val >= zoom_lo) & (y_val <= zoom_hi)
        & (val_predictions >= zoom_lo) & (val_predictions <= zoom_hi)
    )

    if configuration_source == "fresh_optuna_search":
        source_label = "Optuna-selected configuration, final refit"
    elif configuration_source == "reused_optuna_config":
        source_label = "Reused Optuna configuration, final refit"
    elif configuration_source == "fixed_defaults":
        source_label = "Fixed default configuration, final fit"
    else:
        source_label = "Final MLP refit"
    if model_config:
        config_label = (
            f"layers={tuple(model_config['hidden_sizes'])}, "
            f"lr={model_config['lr']:.3g}, "
            f"weight decay={model_config['weight_decay']:.3g}, "
            f"batch size={model_config['batch_size']}"
        )
    else:
        config_label = "configuration not supplied"

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    views = (
        (axes[0], np.ones(len(y_train), dtype=bool), np.ones(len(y_val), dtype=bool), lo, hi,
         "Full target range (all observations)"),
        (axes[1], train_in_zoom, val_in_zoom, zoom_lo, zoom_hi,
         "Central range (zoom; clipped counts shown)"),
    )
    for ax, train_mask, val_mask, axis_lo, axis_hi, title in views:
        ax.scatter(
            y_train[train_mask],
            train_predictions[train_mask],
            color="#1f77b4",
            alpha=0.18 if ax is axes[0] else 0.28,
            s=8 if ax is axes[0] else 10,
            edgecolors="none",
            rasterized=True,
            label=(
                f"Training shown={int(train_mask.sum())}/{len(y_train)}; "
                f"RMSE={train_metrics['RMSE']:.3g}, R²={train_metrics['R2']:.3f}"
            ),
        )
        ax.scatter(
            y_val[val_mask],
            val_predictions[val_mask],
            color="#ff7f0e",
            alpha=0.75,
            s=15,
            edgecolors="none",
            rasterized=True,
            label=(
                f"Validation shown={int(val_mask.sum())}/{len(y_val)}; "
                f"RMSE={val_metrics['RMSE']:.3g}, R²={val_metrics['R2']:.3f}"
            ),
        )
        ax.plot(
            [axis_lo, axis_hi], [axis_lo, axis_hi],
            "k--", linewidth=1.3,
            label="1:1 reference (actual = predicted)",
        )
        ax.set_xlim(axis_lo, axis_hi)
        ax.set_ylim(axis_lo, axis_hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Actual target (original units)")
        ax.set_ylabel("Predicted target (original units)")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="best")

    subtitle = f"{source_label}\n{config_label}"
    if comparison_label:
        subtitle = f"{subtitle}\n{comparison_label}"
    fig.suptitle(subtitle, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    if save_path:
        fig.savefig(save_path, dpi=160)
    plt.show()


def plot_model_comparison(y_true, predictions, save_path=None, title_suffix=None):
    """Compare model predictions against actual targets on the same axes.

    Args:
        y_true: Actual target values in original units.
        predictions: Mapping of model names to predictions in original units.
        save_path: Optional path at which to save the figure.
    """
    actual = np.asarray(y_true)
    all_values = np.concatenate([actual, *(np.asarray(pred) for pred in predictions.values())])
    lo = float(np.min(all_values))
    hi = float(np.max(all_values))
    fig, ax = plt.subplots(figsize=(8, 7))
    for name, y_pred in predictions.items():
        metrics = regression_metrics(actual, y_pred)
        ax.scatter(
            actual,
            y_pred,
            alpha=0.5,
            s=20,
            label=f"{name} (R²={metrics['R2']:.3f}, RMSE={metrics['RMSE']:.3g})",
        )
    ax.plot(
        [lo, hi], [lo, hi], "k--", linewidth=1.5,
        label="1:1 reference (actual = predicted)",
    )
    ax.set_xlabel("Actual target")
    ax.set_ylabel("Predicted target")
    title = "Linear regression vs MLP: held-out test predictions"
    if title_suffix:
        title = f"{title}\n{title_suffix}"
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=160)
    plt.show()


def plot_residuals(y_true, y_pred, save_path=None, title_suffix=None):
    """Plot residuals against predictions, with a horizontal zero reference.

    Residuals are defined as actual target minus predicted target.

    Args:
        y_true: Actual target values in original units.
        y_pred: Predicted target values in original units.
        save_path: Optional path at which to save the figure.
    """
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_pred, residuals, alpha=0.55, s=18)
    ax.axhline(0, color="black", linestyle="--", linewidth=1.3)
    ax.set_xlabel("Predicted target")
    ax.set_ylabel("Residual (actual - predicted)")
    title = "Residuals vs predicted values"
    if title_suffix:
        title = f"{title}\n{title_suffix}"
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=160)
    plt.show()


def plot_runtime_breakdown(stage_timings, total_runtime, save_path=None, title_suffix=None):
    """Plot elapsed time by pipeline stage alongside the total run time.

    Args:
        stage_timings: Mapping from stage names to elapsed seconds.
        total_runtime: Elapsed pipeline seconds before final report-file writing.
        save_path: Optional path at which to save the figure.
    """
    excluded_keys = {"elapsed_before_metrics_json_write"}
    stages = [
        (name, float(seconds))
        for name, seconds in stage_timings.items()
        if name not in excluded_keys and float(seconds) > 0
    ]
    labels = [name.replace("_", " ").title() for name, _ in stages]
    values = [seconds for _, seconds in stages]

    fig, (stage_ax, total_ax) = plt.subplots(
        1,
        2,
        figsize=(13, max(5, 0.42 * len(stages))),
        gridspec_kw={"width_ratios": [4, 1]},
    )
    positions = np.arange(len(stages))
    bars = stage_ax.barh(positions, values, color="#4c78a8")
    stage_ax.set_yticks(positions)
    stage_ax.set_yticklabels(labels)
    stage_ax.invert_yaxis()
    stage_ax.set_xlabel("Elapsed time (seconds)")
    stage_ax.set_title("Runtime by pipeline stage")
    stage_ax.grid(True, axis="x", alpha=0.25)
    stage_ax.bar_label(bars, labels=[f"{value:.2f}s" for value in values], padding=3)

    total_bar = total_ax.barh(["Total"], [total_runtime], color="#f58518")
    total_ax.set_xlim(0, max(total_runtime * 1.15, 1.0))
    total_ax.set_xlabel("Seconds")
    total_ax.set_title("Total run")
    total_ax.grid(True, axis="x", alpha=0.25)
    total_ax.bar_label(total_bar, labels=[f"{total_runtime:.1f}s"], padding=3)
    title = "Experiment runtime (total excludes this chart and montage)"
    if title_suffix:
        title = f"{title}\n{title_suffix}"
    fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=160)
    plt.close(fig)


def create_plot_montage(plot_items, save_path, columns=2):
    """Combine saved plot images into a labeled montage for quick review.

    Args:
        plot_items: Iterable of (image path, title) pairs.
        save_path: Output image path.
        columns: Number of montage columns.
    """
    plot_items = [(path, title) for path, title in plot_items if path.is_file()]
    if not plot_items:
        raise ValueError("No plot images were available to create a montage.")
    rows = int(np.ceil(len(plot_items) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(14, 4.5 * rows),
        squeeze=False,
    )
    for ax, (image_path, title) in zip(axes.flat, plot_items):
        ax.imshow(plt.imread(image_path))
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    for ax in list(axes.flat)[len(plot_items):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=140)
    plt.close(fig)



def run_architecture_sweep(
    X_train,
    y_train_scaled,
    X_val,
    y_val_scaled,
    y_scaler,
    architectures=None,
    epochs=500,
    patience=50,
    min_delta=1e-6,
    grad_clip=1.0,
    seed=42,
):
    """Compare MLP architectures using validation MSE, not test data.

    Args:
        X_train (array-like): Training features scaled using training-only statistics.
        y_train_scaled (array-like): Scaled training targets.
        X_val (array-like): Validation features transformed with the training scaler.
        y_val_scaled (array-like): Validation targets transformed with the training scaler.
        y_scaler (StandardScaler): Fitted target scaler used for original-unit metrics.
        architectures (iterable[tuple[int, ...]] or None): Hidden-layer widths to compare.
        epochs (int): Maximum epochs per architecture.
        patience (int): Early-stopping patience measured in epochs.
        min_delta (float): Minimum scaled validation-MSE improvement required.
        grad_clip (float or None): Maximum gradient norm, or None to disable clipping.
        seed (int): Base random seed, incremented for each architecture.

    Returns:
        pandas.DataFrame: Results sorted by validation MSE, including train/validation
        errors and the best epoch for each architecture.
    """
    from sklearn.metrics import mean_squared_error

    if architectures is None:
        architectures = [
            (32,), (64,), (128,),
            (32, 32), (64, 32), (64, 64), (128, 64),
            (32, 32, 32), (64, 32, 16), (128, 64, 32),
        ]
    architectures = list(architectures)

    y_val_original = y_scaler.inverse_transform(
        np.asarray(y_val_scaled).reshape(-1, 1)
    ).ravel()
    rows = []
    for index, architecture in enumerate(architectures):
        architecture_started = perf_counter()
        model, history = train_optimized_mlp(
            X_train, y_train_scaled, X_val, y_val_scaled,
            hidden_sizes=architecture,
            epochs=epochs,
            patience=patience,
            min_delta=min_delta,
            grad_clip=grad_clip,
            seed=seed + index,
        )
        train_pred = evaluate_pytorch_model(model, X_train, y_scaler)
        val_pred = evaluate_pytorch_model(model, X_val, y_scaler)
        y_train_original = y_scaler.inverse_transform(
            np.asarray(y_train_scaled).reshape(-1, 1)
        ).ravel()
        rows.append({
            "architecture": tuple(architecture),
            "Arch_Str": str(tuple(architecture)),
            "train_mse": float(mean_squared_error(y_train_original, train_pred)),
            "val_mse": float(mean_squared_error(y_val_original, val_pred)),
            "best_epoch": history["best_epoch"],
            "runtime_seconds": perf_counter() - architecture_started,
        })
        print(
            f"Architecture {index + 1}/{len(architectures)} {tuple(architecture)}: "
            f"validation MSE={rows[-1]['val_mse']:.6g}, "
            f"runtime={rows[-1]['runtime_seconds']:.2f} s"
        )
    return pd.DataFrame(rows).sort_values("val_mse").reset_index(drop=True)


def plot_sweep_results(results_df, save_path=None, title_suffix=None):
    """Compare training and validation MSE across tested architectures.

    Args:
        results_df: DataFrame returned by run_architecture_sweep.
        save_path: Optional path at which to save the figure.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    positions = np.arange(len(results_df))
    width = 0.38
    ax.bar(positions - width / 2, results_df["train_mse"], width, label="Training MSE")
    ax.bar(positions + width / 2, results_df["val_mse"], width, label="Validation MSE")
    ax.set_xticks(positions)
    ax.set_xticklabels(results_df["Arch_Str"], rotation=45, ha="right")
    positive = np.concatenate([
        results_df["train_mse"].to_numpy(),
        results_df["val_mse"].to_numpy(),
    ])
    if np.all(positive > 0):
        ax.set_yscale("log")
    ax.set_ylabel("MSE (original target units squared)")
    ax.set_xlabel("Hidden-layer architecture")
    title = "MLP architecture sweep: training vs validation error"
    if title_suffix:
        title = f"{title}\n{title_suffix}"
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=160)
    plt.show()
