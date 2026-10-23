"""Run baseline comparisons, optional Optuna tuning, and final test evaluation."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter

from .data_prep import load_and_preprocess_data
from .hyperparameter_tuning import run_bayesian_search, best_model_config, plot_optuna_convergence
from .linear_regression import MeanBaseline, train_linear_regression
from .neural_network import (
    train_optimized_mlp,
    evaluate_pytorch_model,
    regression_metrics,
    plot_training_history,
    plot_predictions,
    plot_model_comparison,
    plot_residuals,
    plot_gradient_norms,
    plot_train_validation_predictions,
    plot_runtime_breakdown,
    create_plot_montage,
    run_architecture_sweep,
    plot_sweep_results,
)


def _finish_stage(name, started_at, timings):
    """Record and print elapsed wall-clock time for one pipeline stage."""
    elapsed = perf_counter() - started_at
    timings[name] = elapsed
    print(f"Runtime — {name.replace('_', ' ')}: {elapsed:.2f} s")
    return elapsed


def _load_reusable_config(config_path, expected_context):
    """Load an Optuna configuration and reject incompatible experiment context."""
    try:
        with Path(config_path).open("r", encoding="utf-8") as config_file:
            saved = json.load(config_file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read reusable configuration {config_path}: {error}") from error

    if not isinstance(saved, dict):
        raise ValueError("Reusable configuration must contain a JSON object.")
    if saved.get("schema_version") != 1:
        raise ValueError("Reusable configuration has an unsupported schema version.")
    if saved.get("source") != "Optuna":
        raise ValueError("Reusable configuration was not produced by Optuna.")

    saved_context = saved.get("context")
    if not isinstance(saved_context, dict):
        raise ValueError("Reusable configuration is missing its experiment context.")
    # Reuse is valid only for the same data/split context; preventing silent
    # mismatch is more important than accepting every old hyperparameter file.
    mismatches = [
        key for key, expected_value in expected_context.items()
        if saved_context.get(key) != expected_value
    ]
    if mismatches:
        raise ValueError(
            "Reusable configuration does not match this run for: "
            + ", ".join(mismatches)
            + ". Re-optimize or provide a configuration from the matching experiment."
        )

    config = saved.get("configuration")
    if not isinstance(config, dict):
        raise ValueError("Reusable configuration is missing model settings.")
    try:
        raw_hidden_sizes = config["hidden_sizes"]
        # In Python bool is an int subclass, but true/false are not meaningful
        # layer widths or batch sizes in a saved model configuration.
        if not isinstance(raw_hidden_sizes, list) or any(
            not isinstance(width, int) or isinstance(width, bool)
            for width in raw_hidden_sizes
        ):
            raise ValueError
        if not isinstance(config["batch_size"], int) or isinstance(
            config["batch_size"], bool
        ):
            raise ValueError
        hidden_sizes = tuple(raw_hidden_sizes)
        lr = float(config["lr"])
        weight_decay = float(config["weight_decay"])
        batch_size = config["batch_size"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Reusable configuration contains invalid model settings.") from error
    if (
        not hidden_sizes
        or any(width < 1 for width in hidden_sizes)
        or not math.isfinite(lr)
        or lr <= 0
        or not math.isfinite(weight_decay)
        or weight_decay < 0
        or batch_size < 1
    ):
        raise ValueError("Reusable configuration contains out-of-range model settings.")
    return {
        "hidden_sizes": hidden_sizes,
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
    }, saved


def main():
    """Parse command-line options and run the complete regression workflow.

    The workflow loads and splits the dataset, fits baseline models, optionally
    tunes MLP hyperparameters and/or sweeps architectures, trains the selected
    MLP, evaluates final models on the held-out test set, and writes metrics and
    diagnostic plots to the output directory. The test set is not used to
    select model configurations.

    Command-line options control the dataset path, target row, Optuna trials,
    training limits, gradient clipping, random seed, and output location.
    """
    # main.py is nested under src/mma3001; resolve defaults from the repository
    # root so the CLI behaves the same from any current working directory.
    project_dir = Path(__file__).resolve().parents[2]
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
        help="Zero-based row of coefStatMatrix (0=wave resistance Cw at Fr=0.25 per included dataset metadata).",
    )
    parser.add_argument("--trials", type=int, default=20, help="Number of Optuna trials.")
    parser.add_argument(
        "--epochs", type=int, default=1000,
        help="Maximum training epochs per model (early stopping may stop sooner).",
    )
    parser.add_argument(
        "--patience", type=int, default=100,
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
    parser.add_argument(
        "--reuse-config",
        type=Path,
        default=None,
        help="Reuse a saved Optuna configuration JSON instead of running a new search.",
    )
    parser.add_argument("--sweep", action="store_true", help="Run an additional architecture sweep (extra compute).")
    parser.add_argument(
        "--exclude-training-sample", type=int, action="append", default=[],
        help="Zero-based source sample index to omit from training (repeatable); split membership is otherwise fixed.",
    )
    parser.add_argument(
        "--comparison-label", default=None,
        help="Short condition label added to plot titles and run metadata.",
    )
    parser.add_argument("--output-dir", default="outputs", help="Directory for plots and metrics JSON.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    pipeline_started = perf_counter()
    timings = {}

    if args.epochs < 1:
        parser.error("--epochs must be at least 1.")
    if args.patience < 1:
        parser.error("--patience must be at least 1.")
    if args.trials < 1:
        parser.error("--trials must be at least 1.")
    if not math.isfinite(args.min_delta) or args.min_delta < 0:
        parser.error("--min-delta must be finite and non-negative.")
    if not math.isfinite(args.grad_clip) or args.grad_clip < 0:
        parser.error("--grad-clip must be finite and non-negative.")
    if args.skip_tuning and args.reuse_config is not None:
        parser.error("--skip-tuning and --reuse-config cannot be used together.")
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

    stage_started = perf_counter()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _finish_stage("output_directory_setup", stage_started, timings)

    stage_started = perf_counter()
    data = load_and_preprocess_data(
        args.data,
        target_row=args.target_row,
        random_state=args.seed,
        exclude_training_sample_indices=args.exclude_training_sample,
    )
    _finish_stage("data_loading_and_preprocessing", stage_started, timings)
    print(f"Dataset: {args.data.resolve()}")
    print(f"Samples: {data['n_samples']} | features: {data['n_features']}")
    print(f"Split sizes: {data['split_sizes']}")
    if data["excluded_training_sample_indices"]:
        print(
            "Excluded from training (zero-based source sample indices): "
            f"{data['excluded_training_sample_indices']}"
        )
    comparison_label = args.comparison_label or (
        f"Training sample(s) excluded: {data['excluded_training_sample_indices']}"
        if data["excluded_training_sample_indices"]
        else "All source samples retained in their assigned splits"
    )
    target_definition = data["target_definition"]
    if target_definition:
        print(
            f"Target row: {data['target_row']} ({target_definition['quantity']}, "
            f"Fr={target_definition['froude_number']:.2f}, {target_definition['scale']}; "
            "per included dataset metadata)"
        )
    else:
        print(f"Target row: {data['target_row']} (no definition recorded in local dataset metadata)")
    print("All reported metrics below are in the original target units.")

    X_train, X_val, X_test = data["X_train"], data["X_val"], data["X_test"]
    y_train, y_val, y_test = data["y_train"], data["y_val"], data["y_test"]
    # These quick file attributes reject most stale cached configs without
    # hashing a potentially gigabyte-sized MAT file; they are not a checksum.
    dataset_stat = args.data.stat()
    config_context = {
        "data_path": str(args.data.resolve()),
        "data_file_size_bytes": dataset_stat.st_size,
        "data_file_modified_ns": dataset_stat.st_mtime_ns,
        "target_row": data["target_row"],
        "seed": args.seed,
        "n_samples": data["n_samples"],
        "n_features": data["n_features"],
        "split_fractions": {"train": 0.70, "validation": 0.10, "test": 0.20},
        "excluded_training_sample_indices": data["excluded_training_sample_indices"],
    }

    # Optional architecture sweep: select by validation performance only.
    if args.sweep:
        stage_started = perf_counter()
        sweep_df = run_architecture_sweep(
            X_train, data["y_train_scaled"], X_val, data["y_val_scaled"],
            data["y_scaler"], epochs=args.epochs, patience=args.patience,
            min_delta=args.min_delta, grad_clip=grad_clip, seed=args.seed,
        )
        sweep_df.to_csv(output_dir / "architecture_sweep.csv", index=False)
        plot_sweep_results(
            sweep_df, output_dir / "architecture_sweep.png", comparison_label
        )
        print("Best sweep architecture:", sweep_df.iloc[0]["architecture"])
        _finish_stage("architecture_sweep_and_outputs", stage_started, timings)
    else:
        timings["architecture_sweep_and_outputs"] = 0.0
        print("Runtime — architecture sweep: skipped")

    # Baselines: constant mean and linear regression.
    baselines = {}
    results = {}
    # Baselines learn only from the training partition; validation is used for
    # comparison and the held-out test partition is kept untouched for later.
    baseline_factories = (
        ("Mean baseline", lambda: MeanBaseline().fit(X_train, y_train)),
        ("Linear regression", lambda: train_linear_regression(X_train, y_train)),
    )
    for name, fit_model in baseline_factories:
        stage_started = perf_counter()
        model = fit_model()
        baselines[name] = model
        val_pred = model.predict(X_val)
        results[name] = {
            "validation": regression_metrics(y_val, val_pred),
            "test": None,
        }
        print(f"{name} validation: {results[name]['validation']}")
        stage_key = name.lower().replace(" ", "_")
        _finish_stage(f"{stage_key}_fit_and_validation", stage_started, timings)

    config_source = "fixed_defaults"
    reused_config_metadata = None
    # Exactly one source supplies the MLP settings: a validated cache, fixed
    # defaults, or a fresh Optuna study. The optional sweep is separate.
    if args.reuse_config is not None:
        stage_started = perf_counter()
        try:
            config, reused_config_metadata = _load_reusable_config(
                args.reuse_config, config_context
            )
        except ValueError as error:
            parser.error(str(error))
        study = None
        config_source = "reused_optuna_config"
        timings["optuna_search"] = 0.0
        timings["optuna_convergence_plot"] = 0.0
        _finish_stage("reusable_config_loading", stage_started, timings)
        print(f"Reusing Optuna configuration from: {args.reuse_config.resolve()}")
    elif args.skip_tuning:
        timings["optuna_search"] = 0.0
        timings["optuna_convergence_plot"] = 0.0
        print("Runtime — Optuna search: skipped")
        config = {
            "hidden_sizes": (64, 32),
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 64,
        }
        study = None
    else:
        stage_started = perf_counter()
        study = run_bayesian_search(
            X_train, data["y_train_scaled"],
            X_val, data["y_val_scaled"], data["y_scaler"],
            n_trials=args.trials, max_epochs=args.epochs,
            patience=args.patience, min_delta=args.min_delta,
            grad_clip=grad_clip, seed=args.seed,
        )
        _finish_stage("optuna_search", stage_started, timings)
        config = best_model_config(study)
        config_source = "fresh_optuna_search"
        print(f"Best Optuna validation MSE: {study.best_value:.6g}")
        print(f"Best configuration: {config}")
        config_save_started = perf_counter()
        # Save the winner with enough context to detect incompatible reuse on a
        # later run rather than silently applying it to different data/splits.
        saved_config_path = output_dir / "best_config.json"
        with saved_config_path.open("w", encoding="utf-8") as config_file:
            json.dump(
                {
                    "schema_version": 1,
                    "source": "Optuna",
                    "configuration": {
                        **config,
                        "hidden_sizes": list(config["hidden_sizes"]),
                    },
                    "validation_mse": float(study.best_value),
                    "optuna_trials": len(study.trials),
                    "context": config_context,
                    "tuning_settings": {
                        "max_epochs": args.epochs,
                        "patience": args.patience,
                        "min_delta": args.min_delta,
                        "grad_clip": grad_clip,
                    },
                },
                config_file,
                indent=2,
            )
        _finish_stage("best_config_saving", config_save_started, timings)
        print(f"Saved reusable Optuna configuration to: {saved_config_path.resolve()}")
        stage_started = perf_counter()
        plot_optuna_convergence(
            study, output_dir / "optuna_convergence.png", comparison_label
        )
        _finish_stage("optuna_convergence_plot", stage_started, timings)

    stage_started = perf_counter()
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
    _finish_stage("final_mlp_training", stage_started, timings)

    stage_started = perf_counter()
    plot_training_history(
        history, output_dir / "training_loss.png", comparison_label
    )
    plot_gradient_norms(
        history, output_dir / "gradient_norms.png", comparison_label
    )
    # These train/validation predictions come from the final selected-model
    # fit, after Optuna selection or config reuse, rather than a search trial.
    mlp_train_pred = evaluate_pytorch_model(model, X_train, data["y_scaler"])
    mlp_val_pred = evaluate_pytorch_model(model, X_val, data["y_scaler"])
    # This diagnostic uses predictions from the final MLP refit with the
    # selected Optuna configuration (fresh or reused), not a trial's initial model.
    plot_train_validation_predictions(
        y_train,
        mlp_train_pred,
        y_val,
        mlp_val_pred,
        save_path=output_dir / "train_validation_predictions.png",
        model_config=config,
        configuration_source=config_source,
        comparison_label=comparison_label,
    )
    _finish_stage("training_diagnostic_plots", stage_started, timings)

    # Keep test data out of model selection; evaluate each final model here.
    stage_started = perf_counter()
    # Test metrics are calculated only after configurations and checkpoints
    # have been selected using training/validation data.
    test_predictions = {}
    for name, baseline in baselines.items():
        train_pred = baseline.predict(X_train)
        val_pred = baseline.predict(X_val)
        test_predictions[name] = baseline.predict(X_test)
        results[name]["train"] = regression_metrics(y_train, train_pred)
        results[name]["validation"] = regression_metrics(y_val, val_pred)
        results[name]["test"] = regression_metrics(y_test, test_predictions[name])
        if name == "Linear regression":
            plot_predictions(
                y_train,
                train_pred,
                f"Linear regression: training predictions\n{comparison_label}",
                output_dir / "linear_regression_training_predictions.png",
            )
            plot_predictions(
                y_val,
                val_pred,
                f"Linear regression: validation predictions\n{comparison_label}",
                output_dir / "linear_regression_validation_predictions.png",
            )
    mlp_test_pred = evaluate_pytorch_model(model, X_test, data["y_scaler"])
    if config_source == "fresh_optuna_search":
        mlp_name = "Tuned PyTorch MLP"
    elif config_source == "reused_optuna_config":
        mlp_name = "Reused Optuna MLP"
    else:
        mlp_name = "Fixed PyTorch MLP"
    test_predictions[mlp_name] = mlp_test_pred
    results[mlp_name] = {
        "train": regression_metrics(y_train, mlp_train_pred),
        "validation": regression_metrics(y_val, mlp_val_pred),
        "test": regression_metrics(y_test, mlp_test_pred),
        "configuration": {
            "hidden_sizes": list(config["hidden_sizes"]),
            "lr": config["lr"],
            "weight_decay": config["weight_decay"],
            "batch_size": config["batch_size"],
            "best_epoch": history["best_epoch"],
            "stopped_epoch": history["stopped_epoch"],
            "best_validation_loss_scaled": history["best_val_loss_scaled"],
        },
    }
    _finish_stage("held_out_evaluation_and_metrics", stage_started, timings)

    print("\nFINAL HELD-OUT TEST RESULTS")
    for name, metrics in results.items():
        print(f"{name}: {metrics['test']}")

    stage_started = perf_counter()
    plot_predictions(
        y_test, mlp_test_pred,
        f"MLP: held-out test predictions\n{comparison_label}",
        output_dir / "test_predictions.png",
    )
    plot_predictions(
        y_test,
        test_predictions["Linear regression"],
        f"Linear regression: held-out test predictions\n{comparison_label}",
        output_dir / "linear_regression_predictions.png",
    )
    plot_model_comparison(
        y_test,
        {
            "Linear regression": test_predictions["Linear regression"],
            mlp_name: mlp_test_pred,
        },
        output_dir / "test_model_comparison.png",
        title_suffix=comparison_label,
    )
    plot_residuals(
        y_test, mlp_test_pred, output_dir / "test_residuals.png",
        title_suffix=comparison_label,
    )
    _finish_stage("test_diagnostic_plots", stage_started, timings)

    runtime_plot_path = output_dir / "runtime_breakdown.png"
    # The runtime chart reports all measured stages so far; chart/montage
    # rendering and the final JSON write are excluded to avoid self-reference.
    runtime_total_before_summary = perf_counter() - pipeline_started
    runtime_plot_started = perf_counter()
    plot_runtime_breakdown(
        timings, runtime_total_before_summary, runtime_plot_path,
        title_suffix=comparison_label,
    )
    _finish_stage("runtime_breakdown_plot", runtime_plot_started, timings)

    montage_items = [
        (output_dir / "training_loss.png", f"MLP training loss — {comparison_label}"),
        (output_dir / "optuna_convergence.png", f"Optuna convergence — {comparison_label}"),
        (output_dir / "architecture_sweep.png", f"Architecture sweep — {comparison_label}"),
        (output_dir / "gradient_norms.png", f"Gradient norms — {comparison_label}"),
        (output_dir / "train_validation_predictions.png", f"MLP train/validation — {comparison_label}"),
        (output_dir / "test_predictions.png", f"MLP test — {comparison_label}"),
        (output_dir / "linear_regression_predictions.png", f"Linear regression test — {comparison_label}"),
        (output_dir / "linear_regression_training_predictions.png", f"Linear regression train — {comparison_label}"),
        (output_dir / "linear_regression_validation_predictions.png", f"Linear regression validation — {comparison_label}"),
        (output_dir / "test_model_comparison.png", f"Linear regression vs MLP — {comparison_label}"),
        (output_dir / "test_residuals.png", f"MLP test residuals — {comparison_label}"),
        (runtime_plot_path, f"Runtime — {comparison_label}"),
    ]
    montage_started = perf_counter()
    montage_path = output_dir / "plots_montage.png"
    create_plot_montage(montage_items, montage_path)
    _finish_stage("plots_montage_generation", montage_started, timings)

    # JSON serialization time is reported in the terminal after saving; the
    # JSON records elapsed runtime up to the start of its own write.
    # Persist timing values before JSON serialization; the write's own duration
    # is printed afterward because it cannot be contained in the file itself.
    timings["elapsed_before_metrics_json_write"] = perf_counter() - pipeline_started
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "data_path": str(args.data.resolve()),
                "output_dir": str(output_dir.resolve()),
                "plots_montage_path": str(montage_path.resolve()),
                "runtime_breakdown_plot_path": str(runtime_plot_path.resolve()),
                "command": [sys.executable, "-m", "mma3001", *sys.argv[1:]],
                "python_version": sys.version.split()[0],
                "runtime_environment": {
                    name: os.environ.get(name)
                    for name in (
                        "OMP_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                    )
                },
                "target_row": data["target_row"],
                "target_definition": target_definition,
                "target_row_definition_verified": target_definition is not None,
                "split_method": "random sample-level split",
                "split_fractions": {
                    "train": 0.70,
                    "validation": 0.10,
                    "test": 0.20,
                },
                "seed": args.seed,
                "n_samples": data["n_samples"],
                "n_features": data["n_features"],
                "split_sizes": data["split_sizes"],
                "comparison_label": comparison_label,
                "excluded_training_sample_indices": data["excluded_training_sample_indices"],
                "duplicate_audit": data["duplicate_audit"],
                "optuna_trials": len(study.trials) if study else 0,
                "training_settings": {
                    "skip_tuning": args.skip_tuning,
                    "trials_requested": args.trials if study else None,
                    "configuration_source": config_source,
                    "reused_config_path": (
                        str(args.reuse_config.resolve())
                        if args.reuse_config is not None else None
                    ),
                    "reused_config_validation_mse": (
                        reused_config_metadata.get("validation_mse")
                        if reused_config_metadata is not None else None
                    ),
                    "max_epochs": args.epochs,
                    "patience": args.patience,
                    "min_delta": args.min_delta,
                    "grad_clip": grad_clip,
                    "architecture_sweep": args.sweep,
                },
                "runtime_seconds": timings,
                "optuna_trial_runtime_seconds": (
                    [trial.user_attrs.get("runtime_seconds") for trial in study.trials]
                    if study else []
                ),
                "results": results,
            },
            f, indent=2,
        )
    metrics_write_seconds = perf_counter() - (pipeline_started + timings["elapsed_before_metrics_json_write"])
    print(f"\nSaved plots and metrics to: {output_dir.resolve()}")
    print(f"Runtime — metrics JSON write: {metrics_write_seconds:.2f} s")
    print(f"Runtime — total: {perf_counter() - pipeline_started:.2f} s")


if __name__ == "__main__":
    main()
