"""Optuna search using the same PyTorch model and validation procedure."""
import optuna

from neural_network import train_optimized_mlp, evaluate_pytorch_model
from sklearn.metrics import mean_squared_error


def run_bayesian_search(
    X_train,
    y_train_scaled,
    X_val,
    y_val_scaled,
    y_scaler,
    n_trials=20,
    max_epochs=500,
    patience=50,
    min_delta=1e-6,
    grad_clip=1.0,
    seed=42,
):
    """Search MLP hyperparameters using validation MSE in original target units.

    Args:
        X_train (array-like): Training features scaled using training-only statistics.
        y_train_scaled (array-like): Scaled training targets.
        X_val (array-like): Validation features transformed using the training scaler.
        y_val_scaled (array-like): Validation targets transformed using the training scaler.
        y_scaler (StandardScaler): Fitted scaler used for original-unit metrics.
        n_trials (int): Number of Optuna configurations to evaluate.
        max_epochs (int): Maximum training epochs per trial.
        patience (int): Early-stopping patience per trial.
        min_delta (float): Minimum scaled validation-MSE improvement to reset patience.
        grad_clip (float or None): Maximum global gradient norm; None disables clipping.
        seed (int): Seed for Optuna sampling and model training.

    Returns:
        optuna.Study: Completed study; inspect study.best_trial for its winner.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        """Train one suggested configuration and return its validation MSE.

        Args:
            trial: Optuna trial used to sample hyperparameters.

        Returns:
            Validation MSE calculated in the target's original units.
        """
        n_layers = trial.suggest_int("n_layers", 1, 3)
        hidden_sizes = tuple(
            trial.suggest_categorical(f"width_{i}", [32, 64, 96, 128])
            for i in range(n_layers)
        )
        lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

        model, history = train_optimized_mlp(
            X_train, y_train_scaled, X_val, y_val_scaled,
            hidden_sizes=hidden_sizes,
            epochs=max_epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
            min_delta=min_delta,
            grad_clip=grad_clip,
            seed=seed,
        )
        pred = evaluate_pytorch_model(model, X_val, y_scaler)
        y_val_original = y_scaler.inverse_transform(y_val_scaled.reshape(-1, 1)).ravel()
        trial.set_user_attr("hidden_sizes", list(hidden_sizes))
        trial.set_user_attr("best_epoch", history["best_epoch"])
        return mean_squared_error(y_val_original, pred)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    return study


def best_model_config(study):
    """Convert the best Optuna trial into model-training arguments.

    Args:
        study: Completed study returned by run_bayesian_search.

    Returns:
        Dictionary containing hidden sizes, learning rate, weight decay,
        and batch size for the selected configuration.
    """
    params = study.best_trial.params
    n_layers = params["n_layers"]
    return {
        "hidden_sizes": tuple(params[f"width_{i}"] for i in range(n_layers)),
        "lr": params["lr"],
        "weight_decay": params["weight_decay"],
        "batch_size": params["batch_size"],
    }


def plot_optuna_convergence(study, save_path=None):
    """Plot individual Optuna scores and the best validation score so far.

    Args:
        study: Optuna study containing trial results.
        save_path: Optional path at which to save the figure.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    completed = [t for t in study.trials if t.value is not None]
    trial_numbers = [t.number + 1 for t in completed]
    values = [t.value for t in completed]
    running_best = np.minimum.accumulate(values)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(trial_numbers, values, "o--", alpha=0.65, label="Trial validation MSE")
    ax.step(trial_numbers, running_best, where="post", linewidth=2, label="Best so far")
    ax.set_xlabel("Optuna trial")
    ax.set_ylabel("Validation MSE (original target units squared)")
    ax.set_title("Hyperparameter-search history")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=160)
    plt.show()
