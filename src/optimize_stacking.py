import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import numpy as np
import joblib
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

def get_model(model_name, params, random_state):
    """Helper function to instantiate a model from its name and params."""
    if 'random_state' not in params:
        params['random_state'] = random_state
    if model_name == "XGBoostClassifier":
        return XGBClassifier(**params)
    elif model_name == "LGBMClassifier":
        return lgb.LGBMClassifier(**params)
    elif model_name == "RandomForestClassifier":
        return RandomForestClassifier(**params)
    elif model_name == "LogisticRegression":
        return LogisticRegression(**params)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def optimize_stacking_meta_model(cfg: DictConfig):
    """
    Finds the optimal hyperparameters for the stacking meta-model.
    """
    # --- 1. Load Data and Generate OOF Predictions ---
    print("1. Generating Out-of-Fold predictions to be used as training data...")
    processed_data_path = hydra.utils.to_absolute_path(cfg.processed_data.dir) + f"/{cfg.processed_data.train_csv}"
    train_df = pd.read_csv(processed_data_path)
    target_col = cfg.base.target_col
    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]

    stacking_cfg = OmegaConf.load(hydra.utils.to_absolute_path("conf/stacking.yaml"))
    skf = StratifiedKFold(n_splits=stacking_cfg.n_folds, shuffle=True, random_state=cfg.base.random_state)
    
    oof_train_preds = np.zeros((len(X), len(stacking_cfg.base_models)))

    for model_idx, model_name in enumerate(stacking_cfg.base_models):
        print(f"   - Getting OOF preds for {model_name}...")
        model_cfg = OmegaConf.load(hydra.utils.to_absolute_path(f"conf/model/{model_name}.yaml"))
        run_cfg = OmegaConf.create({"base": cfg.base, "model": model_cfg})

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold = y.iloc[train_idx]
            smote = SMOTE(random_state=cfg.base.random_state)
            X_train_fold_resampled, y_train_fold_resampled = smote.fit_resample(X_train_fold, y_train_fold)
            model = get_model(run_cfg.model.name, dict(run_cfg.model.params), run_cfg.base.random_state)
            model.fit(X_train_fold_resampled, y_train_fold_resampled)
            val_preds = model.predict_proba(X_val_fold)[:, 1]
            oof_train_preds[val_idx, model_idx] = val_preds

    # --- 2. Define the Optuna Objective Function for the Meta-Model ---
    meta_model_name_from_config = stacking_cfg.meta_model
    meta_model_cfg = OmegaConf.load(hydra.utils.to_absolute_path(f"conf/model/{meta_model_name_from_config}.yaml"))
    
    def objective(trial: optuna.trial.Trial) -> float:
        """
        Objective function to tune the meta-model's hyperparameters.
        """
        if meta_model_cfg.name == "LogisticRegression":
            params = {
                'C': trial.suggest_float("C", 1e-4, 1e2, log=True),
                'random_state': cfg.base.random_state
            }
            meta_model = LogisticRegression(**params)
        
        elif meta_model_cfg.name == "LGBMClassifier":
            params = {
                'n_estimators': trial.suggest_int("n_estimators", 50, 500),
                'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                'num_leaves': trial.suggest_int("num_leaves", 5, 30),
                'max_depth': trial.suggest_int("max_depth", 2, 5),
                'random_state': cfg.base.random_state
            }
            meta_model = lgb.LGBMClassifier(**params)
        else:
            raise ValueError(f"Meta-model tuning not configured for: {meta_model_cfg.name}")
        
        score = cross_val_score(meta_model, oof_train_preds, y, cv=skf, scoring='roc_auc').mean()
        return score

    # --- 3. Run the Optuna Study ---
    print(f"\n2. Starting optimization for the {meta_model_cfg.name} meta-model...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    print("\nMeta-model optimization finished.")
    print("Best trial:")
    trial = study.best_trial
    print(f"  Best AUC Score: {trial.value}")
    print("  Best Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

if __name__ == "__main__":
    optimize_stacking_meta_model()
