import os
import hydra
from omegaconf import DictConfig
import pandas as pd
import json
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from imblearn.over_sampling import SMOTE

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def train_and_evaluate(cfg: DictConfig):
    """
    Loads split data, handles class imbalance, trains the model,
    evaluates it, and saves artifacts using configuration from Hydra.
    """
    # --- 1. Load Data Splits ---
    split_data_dir = hydra.utils.to_absolute_path(cfg.split_data.dir)
    train_path = os.path.join(split_data_dir, cfg.split_data.train_path)
    test_path = os.path.join(split_data_dir, cfg.split_data.test_path)

    print("1. Loading data splits...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    target_col = cfg.base.target_col
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    # --- 2. Handle Imbalanced Data with SMOTE ---
    print("2. Handling class imbalance with SMOTE...")
    # SMOTE (Synthetic Minority Over-sampling Technique) creates new synthetic
    # samples of the minority class ('yes' or 1) to balance the dataset.
    # This should only be applied to the training data.
    smote = SMOTE(random_state=cfg.base.random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # --- 3. Train the Model ---
    print("3. Training the model...")
    # Convert Hydra's DictConfig to a regular Python dictionary for unpacking
    model_params = dict(cfg.model.params)
    
    model = XGBClassifier(
        random_state=cfg.base.random_state,
        **model_params
    )
    
    model.fit(X_train_resampled, y_train_resampled)

    # --- 4. Evaluate the Model ---
    print("4. Evaluating the model...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    print(f"   AUC Score: {auc:.4f}")

    # --- 5. Save Metrics and Model ---
    metrics_dir = hydra.utils.to_absolute_path(cfg.metrics.dir)
    metrics_path = os.path.join(metrics_dir, cfg.metrics.filename)
    
    model_dir = hydra.utils.to_absolute_path(cfg.model.dir)
    model_path = os.path.join(model_dir, cfg.model.filename)

    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print(f"\n5. Saving metrics to '{metrics_path}'...")
    metrics = {'accuracy': accuracy, 'f1_score': f1, 'auc': auc}
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"   Saving model to '{model_path}'...")
    joblib.dump(model, model_path)

    print("\nTraining and evaluation complete.")

if __name__ == '__main__':
    train_and_evaluate()
