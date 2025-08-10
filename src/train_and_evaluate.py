import os
import yaml
import pandas as pd
import json
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

def train_and_evaluate(config_path):
    """
    Loads split data, handles class imbalance using SMOTE, trains the model,
    evaluates it, and saves the model and metrics.
    """
    # Load configuration from params.yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # --- 1. Load Data Splits ---
    split_data_dir = config['split_data']['dir']
    train_path = os.path.join(split_data_dir, config['split_data']['train_path'])
    test_path = os.path.join(split_data_dir, config['split_data']['test_path'])

    print("1. Loading data splits...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    target_col = 'y'
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    # --- 2. Handle Imbalanced Data with SMOTE (Your Point #5) ---
    print("2. Handling class imbalance with SMOTE...")
    # SMOTE (Synthetic Minority Over-sampling Technique) creates new synthetic
    # samples of the minority class ('yes' or 1) to balance the dataset.
    # This should only be applied to the training data.
    smote = SMOTE(random_state=config['base']['random_state'])
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"   Original training data shape: {X_train.shape}")
    print(f"   Resampled training data shape: {X_train_resampled.shape}")

    # --- 3. Train the Model ---
    print("3. Training the XGBoost model...")
    model_params = config['model']['params']
    random_state = config['base']['random_state']

    model = XGBClassifier(
        random_state=random_state,
        **model_params
    )
    
    model.fit(X_train_resampled, y_train_resampled)

    # --- 4. Evaluate the Model ---
    print("4. Evaluating the model...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] # Probabilities for the positive class

    # Calculate metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    print(f"   AUC Score: {auc:.4f}")

    # --- 5. Save Metrics and Model ---
    metrics_dir = config['metrics']['dir']
    metrics_path = os.path.join(metrics_dir, config['metrics']['filename'])
    
    model_dir = config['model']['dir']
    model_path = os.path.join(model_dir, "model.joblib")

    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print(f"\n5. Saving metrics to '{metrics_path}'...")
    metrics = {
        'accuracy': accuracy,
        'f1_score': f1,
        'auc': auc
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"   Saving model to '{model_path}'...")
    joblib.dump(model, model_path)

    print("\nTraining and evaluation complete.")


if __name__ == '__main__':
    train_and_evaluate(config_path='params.yaml')
