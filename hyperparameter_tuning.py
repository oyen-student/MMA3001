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
    max_epochs=160,
    seed=42,
):
    """Tune architecture and learning rate using validation MSE in original units."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
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
            patience=18,
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
    params = study.best_trial.params
    n_layers = params["n_layers"]
    return {
        "hidden_sizes": tuple(params[f"width_{i}"] for i in range(n_layers)),
        "lr": params["lr"],
        "weight_decay": params["weight_decay"],
        "batch_size": params["batch_size"],
    }


def plot_optuna_convergence(study, save_path=None):
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
