import os
import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import json
import joblib
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from imblearn.over_sampling import SMOTE
import mlflow

def train_and_evaluate(cfg: DictConfig) -> float:
    """
    Loads split data, trains the specified model, evaluates it, and logs 
    everything to MLflow. Returns the primary evaluation metric (AUC).
    """
    with mlflow.start_run():
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))

        # --- 1. Load Data Splits ---
        split_data_dir = hydra.utils.to_absolute_path(cfg.split_data.dir)
        train_path = os.path.join(split_data_dir, cfg.split_data.train_path)
        test_path = os.path.join(split_data_dir, cfg.split_data.test_path)
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        target_col = cfg.base.target_col
        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]

        # --- 2. Handle Imbalanced Data with SMOTE ---
        smote = SMOTE(random_state=cfg.base.random_state)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        # --- 3. Train the Model ---
        print(f"3. Training the {cfg.model.name} model...")
        model_params = dict(cfg.model.params)
        
        if cfg.model.name == "XGBoostClassifier":
            model = XGBClassifier(**model_params)
        elif cfg.model.name == "LGBMClassifier":
            model = lgb.LGBMClassifier(**model_params)
        elif cfg.model.name == "RandomForestClassifier":
            model = RandomForestClassifier(**model_params)
        elif cfg.model.name == "LogisticRegression":
            model = LogisticRegression(**model_params)
        else:
            raise ValueError(f"Unsupported model type: {cfg.model.name}")
        
        model.fit(X_train_resampled, y_train_resampled)

        # --- 4. Evaluate the Model ---
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        print(f"   AUC Score: {auc:.4f}")
        mlflow.log_metric("auc", auc)

        # --- 5. Save Artifacts ---
        mlflow.sklearn.log_model(
            sk_model=model, 
            artifact_path="model",
            registered_model_name=cfg.model.name,
            input_example=X_train_resampled.head()
        )
        return auc

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    train_and_evaluate(cfg)

if __name__ == '__main__':
    main()
