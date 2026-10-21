"""PyTorch MLP regression, early stopping, and evaluation plots."""
from copy import deepcopy
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PyTorchMLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes=(64, 32)):
        super().__init__()
        layers = []
        in_dim = input_dim
        for width in hidden_sizes:
            layers.extend([nn.Linear(in_dim, int(width)), nn.ReLU()])
            in_dim = int(width)
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
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
    min_delta=1e-5,
    seed=42,
):
    """Train using scaled targets and return best-validation model and history."""
    set_seed(seed)
    device = get_device()
    X_train_t = torch.as_tensor(np.asarray(X_train), dtype=torch.float32)
    y_train_t = torch.as_tensor(np.asarray(y_train_scaled), dtype=torch.float32).reshape(-1, 1)
    X_val_t = torch.as_tensor(np.asarray(X_val), dtype=torch.float32, device=device)
    y_val_t = torch.as_tensor(np.asarray(y_val_scaled), dtype=torch.float32, device=device)

    loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=min(batch_size, len(X_train_t)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    model = PyTorchMLP(X_train.shape[1], hidden_sizes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = deepcopy(model.state_dict())
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        train_sum = 0.0
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
            train_sum += loss.item() * len(batch_X)

        train_loss = train_sum / len(X_train_t)
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

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
    """Return predictions in the target's original physical units."""
    device = next(model.parameters()).device
    X_t = torch.as_tensor(np.asarray(X_data), dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        pred_scaled = model(X_t).cpu().numpy().reshape(-1, 1)
    return y_scaler.inverse_transform(pred_scaled).ravel()


def regression_metrics(y_true, y_pred):
    return {
        "MSE": float(mean_squared_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def plot_training_history(history, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(1, len(history["train_loss"]) + 1), history["train_loss"], label="Training loss")
    ax.plot(np.arange(1, len(history["val_loss"]) + 1), history["val_loss"], label="Validation loss")
    ax.axvline(history["best_epoch"], linestyle="--", alpha=0.7, label=f"Best epoch: {history['best_epoch']}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (standardised target)")
    ax.set_title("MLP training and validation convergence")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=160)
    plt.show()


def plot_predictions(y_true, y_pred, title="Predicted vs actual", save_path=None):
    metrics = regression_metrics(y_true, y_pred)
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.55, s=18)
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.5)
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


def plot_residuals(y_true, y_pred, save_path=None):
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_pred, residuals, alpha=0.55, s=18)
    ax.axhline(0, color="black", linestyle="--", linewidth=1.3)
    ax.set_xlabel("Predicted target")
    ax.set_ylabel("Residual (actual - predicted)")
    ax.set_title("Residuals vs predicted values")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=160)
    plt.show()



def run_architecture_sweep(
    X_train,
    y_train_scaled,
    X_val,
    y_val_scaled,
    y_scaler,
    architectures=None,
    epochs=100,
    seed=42,
):
    """Compare architectures on validation MSE; does not inspect the test set."""
    from sklearn.metrics import mean_squared_error

    if architectures is None:
        architectures = [
            (32,), (64,), (128,),
            (32, 32), (64, 32), (64, 64), (128, 64),
            (32, 32, 32), (64, 32, 16), (128, 64, 32),
        ]

    y_val_original = y_scaler.inverse_transform(
        np.asarray(y_val_scaled).reshape(-1, 1)
    ).ravel()
    rows = []
    for index, architecture in enumerate(architectures):
        model, history = train_optimized_mlp(
            X_train, y_train_scaled, X_val, y_val_scaled,
            hidden_sizes=architecture,
            epochs=epochs,
            patience=15,
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
        })
    return pd.DataFrame(rows).sort_values("val_mse").reset_index(drop=True)


def plot_sweep_results(results_df, save_path=None):
    """Plot actual architecture-sweep results (not manually entered values)."""
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
    ax.set_title("MLP architecture sweep: training vs validation error")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=160)
    plt.show()
