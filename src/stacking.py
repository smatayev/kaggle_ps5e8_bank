import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE # Import SMOTE

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
def run_stacking(cfg: DictConfig):
    """
    Runs a stacking ensemble using out-of-fold predictions.
    This version uses the main config for paths and applies SMOTE within folds.
    """
    # --- 1. Load Data and Preprocessor using Hydra Config ---
    print("1. Loading data and preprocessor...")
    processed_data_path = hydra.utils.to_absolute_path(cfg.processed_data.dir) + f"/{cfg.processed_data.train_csv}"
    test_data_path = hydra.utils.to_absolute_path(cfg.data_source.raw_dir) + f"/{cfg.data_source.test_csv}"
    preprocessor_path = hydra.utils.to_absolute_path(cfg.preprocessor.dir) + f"/{cfg.preprocessor.filename}"

    train_df = pd.read_csv(processed_data_path)
    test_df_raw = pd.read_csv(test_data_path)
    preprocessor = joblib.load(preprocessor_path)

    target_col = cfg.base.target_col
    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]

    test_df_raw['contacted_previously'] = (test_df_raw['pdays'] != -1).astype(int)
    X_test_processed = preprocessor.transform(test_df_raw)

    # --- 2. Generate Out-of-Fold (OOF) Predictions ---
    print("\n2. Generating Out-of-Fold predictions for base models...")
    stacking_cfg = OmegaConf.load(hydra.utils.to_absolute_path("conf/stacking.yaml"))
    
    skf = StratifiedKFold(n_splits=stacking_cfg.n_folds, shuffle=True, random_state=cfg.base.random_state)
    
    oof_train_preds = np.zeros((len(X), len(stacking_cfg.base_models)))
    oof_test_preds = np.zeros((len(X_test_processed), len(stacking_cfg.base_models)))

    for model_idx, model_name in enumerate(stacking_cfg.base_models):
        print(f"   - Training {model_name}...")
        
        model_specific_cfg = OmegaConf.load(hydra.utils.to_absolute_path(f"conf/model/{model_name}.yaml"))
        
        # FIX: Create a new, clean config for this specific run to avoid struct conflicts
        run_cfg = OmegaConf.create({
            "base": cfg.base,
            "model": model_specific_cfg
        })

        test_preds_for_model = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold = y.iloc[train_idx]

            smote = SMOTE(random_state=cfg.base.random_state)
            X_train_fold_resampled, y_train_fold_resampled = smote.fit_resample(X_train_fold, y_train_fold)

            model = get_model(run_cfg.model.name, dict(run_cfg.model.params), run_cfg.base.random_state)
            model.fit(X_train_fold_resampled, y_train_fold_resampled)
            
            val_preds = model.predict_proba(X_val_fold)[:, 1]
            oof_train_preds[val_idx, model_idx] = val_preds
            
            test_preds_for_model.append(model.predict_proba(X_test_processed)[:, 1])
        
        oof_test_preds[:, model_idx] = np.mean(test_preds_for_model, axis=0)

    # --- 3. Train the Meta-Model ---
    print("\n3. Training the meta-model...")
    meta_model_name = stacking_cfg.meta_model
    meta_model_specific_cfg = OmegaConf.load(hydra.utils.to_absolute_path(f"conf/model/{meta_model_name}.yaml"))
    
    # FIX: Create a clean config for the meta-model as well
    meta_run_cfg = OmegaConf.create({
        "base": cfg.base,
        "model": meta_model_specific_cfg
    })
    
    meta_model = get_model(meta_run_cfg.model.name, dict(meta_run_cfg.model.params), meta_run_cfg.base.random_state)
    
    meta_model.fit(oof_train_preds, y)

    # --- 4. Generate Final Submission ---
    print("\n4. Generating final submission...")
    final_predictions = meta_model.predict_proba(oof_test_preds)[:, 1]
    
    submission_df = pd.DataFrame({'id': test_df_raw['id'], 'y': final_predictions})
    submission_path = "submission_stacking.csv"
    submission_df.to_csv(submission_path, index=False)

    print(f"\nStacking submission file created at '{submission_path}'")
    print(submission_df.head())

if __name__ == "__main__":
    run_stacking()
