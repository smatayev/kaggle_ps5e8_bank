import os
import hydra
from omegaconf import DictConfig
import pandas as pd
import joblib
from xgboost import XGBClassifier
import lightgbm as lgb 
from imblearn.over_sampling import SMOTE

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def train_final_model(cfg: DictConfig):
    """
    Loads the FULL processed dataset, applies SMOTE, trains the model,
    and saves the final version using configuration from Hydra.
    """
    # --- 1. Load Full Processed Data ---
    processed_data_dir = hydra.utils.to_absolute_path(cfg.processed_data.dir)
    train_path = os.path.join(processed_data_dir, cfg.processed_data.train_csv)

    print("1. Loading full processed training data...")
    train_df = pd.read_csv(train_path)

    target_col = cfg.base.target_col
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    # --- 2. Handle Imbalanced Data with SMOTE ---
    print("2. Applying SMOTE to the full dataset...")
    smote = SMOTE(random_state=cfg.base.random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # --- 3. Train the Final Model ---
    print("3. Training the final model on all data...")
    model_params = dict(cfg.model.params)
    
    # Dynamic model selection based on Hydra config
    if cfg.model.name == "XGBoostClassifier":
        final_model = XGBClassifier(**model_params)
    elif cfg.model.name == "LGBMClassifier":
        final_model = lgb.LGBMClassifier(**model_params)
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
