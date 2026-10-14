
import optuna
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

def run_bayesian_search(X_train, y_train_scaled, X_val, y_val, target_scale=1000.0, n_trials=10):
    def objective(trial):
        num_layers = trial.suggest_int('num_layers', 1, 3)
        layers = [trial.suggest_int(f'nodes_l{i}', 32, 128) for i in range(num_layers)]
        
        mlp = MLPRegressor(
            hidden_layer_sizes=tuple(layers),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size=64,
            learning_rate_init=0.001,
            learning_rate='adaptive',
            max_iter=100,
            random_state=42
        )
        
        mlp.fit(X_train, y_train_scaled)
        val_preds = mlp.predict(X_val) / target_scale
        return mean_squared_error(y_val, val_preds)
        
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    return study

def plot_optuna_convergence(study):
    trials = study.trials
    trial_numbers = [t.number for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    validation_mses = [t.value for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    running_minimums = []
    current_min = float('inf')
    for val in validation_mses:
        if val < current_min:
            current_min = val
        running_minimums.append(current_min)
        
    plt.figure(figsize=(12, 5))
    plt.plot(trial_numbers, validation_mses, marker='o', linestyle='--', color='tomato', alpha=0.7, label='Trial Objective (Validation MSE)')
    plt.step(trial_numbers, running_minimums, where='post', color='darkblue', linewidth=2.5, label='Best Value Found')
    
    plt.yscale('log')
    plt.xlabel('Trial Number', fontweight='bold')
    plt.ylabel('Validation MSE (Log Scale)', fontweight='bold')
    plt.title('Optuna Bayesian Optimization Convergence History', fontsize=13, fontweight='bold')
    plt.grid(True, which="both", linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()
