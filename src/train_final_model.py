import os
import hydra
from omegaconf import DictConfig
import pandas as pd
import joblib
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def train_final_model(cfg: DictConfig):
    """
    Loads the FULL processed dataset, applies SMOTE, trains the specified 
    final model, and saves it.
    """
    # --- 1. Load Full Processed Data ---
    processed_data_dir = hydra.utils.to_absolute_path(cfg.processed_data.dir)
    train_path = os.path.join(processed_data_dir, cfg.processed_data.train_csv)
    train_df = pd.read_csv(train_path)
    target_col = cfg.base.target_col
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    # --- 2. Handle Imbalanced Data with SMOTE ---
    smote = SMOTE(random_state=cfg.base.random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # --- 3. Train the Final Model ---
    print(f"3. Training the final {cfg.model.name} model on all data...")
    model_params = dict(cfg.model.params)
    
    if cfg.model.name == "XGBoostClassifier":
        final_model = XGBClassifier(**model_params)
    elif cfg.model.name == "LGBMClassifier":
        final_model = lgb.LGBMClassifier(**model_params)
    elif cfg.model.name == "RandomForestClassifier":
        final_model = RandomForestClassifier(**model_params)
    elif cfg.model.name == "LogisticRegression":
        final_model = LogisticRegression(**model_params)
    else:
        raise ValueError(f"Unsupported model type: {cfg.model.name}")
    
    final_model.fit(X_train_resampled, y_train_resampled)

    # --- 4. Save the Final Model ---
    model_dir = hydra.utils.to_absolute_path(cfg.model.dir)
    final_model_path = os.path.join(model_dir, cfg.model.final_filename)
    os.makedirs(model_dir, exist_ok=True)
    print(f"\n4. Saving final model to '{final_model_path}'...")
    joblib.dump(final_model, final_model_path)

    print("\nFinal model training complete.")

if __name__ == '__main__':
    train_final_model()
