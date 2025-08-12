import os
import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import json
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from imblearn.over_sampling import SMOTE
import mlflow

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def train_and_evaluate(cfg: DictConfig):
    """
    Loads split data, handles class imbalance, trains the model,
    evaluates it, saves artifacts using configuration from Hydra, 
    and logs everything to MLflow.
    """
    # Use an MLflow context manager to automatically log runs
    with mlflow.start_run():
        print("MLflow Run Started...")
        # Log the entire configuration as parameters
        print("Logging parameters to MLflow...")
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))

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

        # Log metrics to MLflow
        print("Logging metrics to MLflow...")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("auc", auc)

        # --- 5. Save Artifacts ---
        # Note: MLflow will save the model in its own format.
        # We can still save our own copy if needed.
        model_dir = hydra.utils.to_absolute_path(cfg.model.dir)
        model_path = os.path.join(model_dir, cfg.model.filename)
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(model, model_path)
        
        # Log the model as an artifact in MLflow
        print("Logging model to MLflow with signature...")
        # Add an input_example to automatically infer the model signature.
        mlflow.sklearn.log_model(
            sk_model=model, 
            artifact_path="model", # This is the sub-directory name within the run
            registered_model_name=cfg.model.name, # Use 'name' for the registry
            input_example=X_train_resampled.head()
        )

        print("\nTraining, evaluation, and logging complete.")

        return auc # <-- Return the score for Optuna

# Add a main entry point for direct script execution
@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    train_and_evaluate(cfg)

if __name__ == '__main__':
    main() # <-- Call the main function
