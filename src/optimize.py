import hydra
from omegaconf import DictConfig
import optuna
from train_and_evaluate import train_and_evaluate

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def optimize(cfg: DictConfig):
    """
    Main function to run the Optuna optimization.
    This function is decorated with Hydra to load the base configuration.
    """

    def objective(trial: optuna.trial.Trial) -> float:
        """
        This is the objective function for Optuna.
        It takes a trial object, suggests hyperparameters, updates the config,
        and runs the training function.
        """
        # Create a mutable copy of the config for this trial
        trial_cfg = cfg.copy()
        
        # Suggest hyperparameters for Optuna to try using the 'trial' object
        trial_cfg.model.params.n_estimators = trial.suggest_int("n_estimators", 100, 1000, step=50)
        trial_cfg.model.params.max_depth = trial.suggest_int("max_depth", 3, 10)
        trial_cfg.model.params.eta = trial.suggest_float("eta", 0.01, 0.3, log=True)
        trial_cfg.model.params.subsample = trial.suggest_float("subsample", 0.5, 1.0)
        trial_cfg.model.params.colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)

        # Run the training and evaluation function with the new config
        try:
            auc_score = train_and_evaluate(trial_cfg)
            return auc_score
        except Exception as e:
            print(f"Trial failed with error: {e}")
            # Tell Optuna to prune the trial if it fails
            raise optuna.exceptions.TrialPruned()

    # Create an Optuna study. We want to maximize the AUC score.
    study = optuna.create_study(direction="maximize")
    
    # Start the optimization process, passing the inner 'objective' function
    study.optimize(objective, n_trials=20)

    print("\nOptimization finished.")
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    optimize()
