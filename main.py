"""Run baseline comparisons, optional Optuna tuning, and final test evaluation."""
import argparse
import json
from pathlib import Path

from data_prep import load_and_preprocess_data
from hyperparameter_tuning import run_bayesian_search, best_model_config, plot_optuna_convergence
from linear_regression import MeanBaseline, train_linear_regression
from neural_network import (
    train_optimized_mlp,
    evaluate_pytorch_model,
    regression_metrics,
    plot_training_history,
    plot_predictions,
    plot_residuals,
    plot_gradient_norms,
    run_architecture_sweep,
    plot_sweep_results,
)


def main():
    project_dir = Path(__file__).resolve().parent
    default_data_path = (
        project_dir / "data" / "dr-benchmark-5415hull" / "DTMB5415-database.mat"
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=default_data_path,
        help=(
            "Path to DTMB5415-database.mat "
            f"(default: {default_data_path})"
        ),
    )
    parser.add_argument(
        "--target-row", type=int, default=0,
        help="Row of coefStatMatrix to predict (DEFAULT 0 MUST be verified from dataset documentation).",
    )
    parser.add_argument("--trials", type=int, default=20, help="Number of Optuna trials.")
    parser.add_argument(
        "--epochs", type=int, default=500,
        help="Maximum training epochs per model (early stopping may stop sooner).",
    )
    parser.add_argument(
        "--patience", type=int, default=50,
        help="Epochs without validation improvement before early stopping.",
    )
    parser.add_argument(
        "--min-delta", type=float, default=1e-6,
        help="Minimum validation-MSE improvement (standardised target) to reset early stopping.",
    )
    parser.add_argument(
        "--grad-clip", type=float, default=1.0,
        help="Maximum global gradient norm; set 0 to disable clipping.",
    )
    parser.add_argument("--skip-tuning", action="store_true", help="Use a fixed modest MLP instead of Optuna.")
    parser.add_argument("--sweep", action="store_true", help="Run an additional architecture sweep (extra compute).")
    parser.add_argument("--output-dir", default="outputs", help="Directory for plots and metrics JSON.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1.")
    if args.patience < 1:
        parser.error("--patience must be at least 1.")
    if args.min_delta < 0:
        parser.error("--min-delta must be non-negative.")
    if args.grad_clip < 0:
        parser.error("--grad-clip must be non-negative.")
    grad_clip = args.grad_clip if args.grad_clip > 0 else None

    if not args.data.is_file():
        raise FileNotFoundError(
            f"MATLAB dataset not found: {args.data}\n\n"
            "Download the dataset from https://zenodo.org/records/19222008, "
            "extract the archive, and move the entire 'dr-benchmark-5415hull' "
            "folder into the repository's 'data' folder. The expected file path is:\n"
            f"  {default_data_path}\n\n"
            "If the file is stored elsewhere, pass its full path using --data."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_and_preprocess_data(args.data, target_row=args.target_row, random_state=args.seed)
    print(f"Dataset: {args.data.resolve()}")
    print(f"Samples: {data['n_samples']} | features: {data['n_features']}")
    print(f"Split sizes: {data['split_sizes']}")
    print(f"Target row: {data['target_row']} (verify its physical meaning in the dataset documentation)")
    print("All reported metrics below are in the original target units.")

    X_train, X_val, X_test = data["X_train"], data["X_val"], data["X_test"]
    y_train, y_val, y_test = data["y_train"], data["y_val"], data["y_test"]

    # Optional architecture sweep: select by validation performance only.
    if args.sweep:
        sweep_df = run_architecture_sweep(
            X_train, data["y_train_scaled"], X_val, data["y_val_scaled"],
            data["y_scaler"], epochs=args.epochs, patience=args.patience,
            min_delta=args.min_delta, grad_clip=grad_clip, seed=args.seed,
        )
        sweep_df.to_csv(output_dir / "architecture_sweep.csv", index=False)
        plot_sweep_results(sweep_df, output_dir / "architecture_sweep.png")
        print("Best sweep architecture:", sweep_df.iloc[0]["architecture"])

    # Baselines: constant mean and linear regression.
    baselines = {
        "Mean baseline": MeanBaseline().fit(X_train, y_train),
        "Linear regression": train_linear_regression(X_train, y_train),
    }
    results = {}
    for name, model in baselines.items():
        val_pred = model.predict(X_val)
        results[name] = {
            "validation": regression_metrics(y_val, val_pred),
            "test": None,
        }
        print(f"{name} validation: {results[name]['validation']}")

    if args.skip_tuning:
        config = {
            "hidden_sizes": (64, 32),
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 64,
        }
        study = None
    else:
        study = run_bayesian_search(
            X_train, data["y_train_scaled"],
            X_val, data["y_val_scaled"], data["y_scaler"],
            n_trials=args.trials, max_epochs=args.epochs,
            patience=args.patience, min_delta=args.min_delta,
            grad_clip=grad_clip, seed=args.seed,
        )
        config = best_model_config(study)
        print(f"Best Optuna validation MSE: {study.best_value:.6g}")
        print(f"Best configuration: {config}")
        plot_optuna_convergence(study, output_dir / "optuna_convergence.png")

    model, history = train_optimized_mlp(
        X_train, data["y_train_scaled"],
        X_val, data["y_val_scaled"],
        hidden_sizes=config["hidden_sizes"],
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        batch_size=config["batch_size"],
        epochs=args.epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        grad_clip=grad_clip,
        seed=args.seed,
    )
    plot_training_history(history, output_dir / "training_validation_loss.png")
    plot_gradient_norms(history, output_dir / "gradient_norms.png")

    # Keep test data out of model selection; evaluate each final model here.
    for name, baseline in baselines.items():
        results[name]["test"] = regression_metrics(y_test, baseline.predict(X_test))
    mlp_test_pred = evaluate_pytorch_model(model, X_test, data["y_scaler"])
    results["Tuned PyTorch MLP" if study else "Fixed PyTorch MLP"] = {
        "validation": regression_metrics(
            y_val, evaluate_pytorch_model(model, X_val, data["y_scaler"])
        ),
        "test": regression_metrics(y_test, mlp_test_pred),
        "configuration": {
            "hidden_sizes": list(config["hidden_sizes"]),
            "lr": config["lr"],
            "weight_decay": config["weight_decay"],
            "batch_size": config["batch_size"],
            "best_epoch": history["best_epoch"],
        },
    }

    print("\nFINAL HELD-OUT TEST RESULTS")
    for name, metrics in results.items():
        print(f"{name}: {metrics['test']}")

    plot_predictions(y_test, mlp_test_pred, "MLP: held-out test predictions", output_dir / "test_predictions.png")
    plot_residuals(y_test, mlp_test_pred, output_dir / "test_residuals.png")

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "target_row": data["target_row"],
                "target_row_definition_verified": False,
                "n_samples": data["n_samples"],
                "n_features": data["n_features"],
                "split_sizes": data["split_sizes"],
                "optuna_trials": len(study.trials) if study else 0,
                "results": results,
            },
            f, indent=2,
        )
    print(f"\nSaved plots and metrics to: {output_dir.resolve()}")
    print("Reminder: confirm coefStatMatrix row definitions before interpreting this as wave-resistance prediction.")


if __name__ == "__main__":
    main()
