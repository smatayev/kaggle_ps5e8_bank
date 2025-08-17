import os
import hydra
from omegaconf import DictConfig
import pandas as pd
import joblib

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def predict_weighted_ensemble(cfg: DictConfig):
    """
    Loads the final trained models, creates a weighted averaged
    ensemble prediction, and generates a submission file.
    """
    # --- 1. Load Artifacts and Raw Test Data ---
    print("1. Loading artifacts and raw test data...")
    model_dir = hydra.utils.to_absolute_path(cfg.model.dir)
    
    # Define paths for all three final models
    xgb_model_path = os.path.join(model_dir, "final_xgb_model.joblib")
    lgbm_model_path = os.path.join(model_dir, "final_lgbm_model.joblib")
    rf_model_path = os.path.join(model_dir, "final_rf_model.joblib")
    
    preprocessor_path = hydra.utils.to_absolute_path(os.path.join(cfg.preprocessor.dir, cfg.preprocessor.filename))
    test_raw_path = hydra.utils.to_absolute_path(os.path.join(cfg.data_source.raw_dir, cfg.data_source.test_csv))

    print("   - Loading models and preprocessor...")
    model_xgb = joblib.load(xgb_model_path)
    model_lgbm = joblib.load(lgbm_model_path)
    model_rf = joblib.load(rf_model_path)
    preprocessor = joblib.load(preprocessor_path)
    df_test_raw = pd.read_csv(test_raw_path)

    if 'id' in df_test_raw.columns:
        test_ids = df_test_raw['id']
    else:
        test_ids = pd.Series(range(len(df_test_raw)), name="id")

    # --- 2. Preprocess the Test Data ---
    print("2. Preprocessing the test data...")
    df_test_raw['contacted_previously'] = (df_test_raw['pdays'] != -1).astype(int)
    X_test_processed = preprocessor.transform(df_test_raw)
    
    # --- 3. Generate Predictions from All Models ---
    print("3. Generating probability predictions from each model...")
    pred_xgb_proba = model_xgb.predict_proba(X_test_processed)[:, 1]
    pred_lgbm_proba = model_lgbm.predict_proba(X_test_processed)[:, 1]
    pred_rf_proba = model_rf.predict_proba(X_test_processed)[:, 1]

    # --- 4. Create Weighted Average Ensemble Prediction ---
    print("4. Applying weighted average to create ensemble...")
    weights = {'xgb': 0.475, 'lgbm': 0.475, 'rf': 0.05}
    final_pred_proba = (weights['xgb'] * pred_xgb_proba + 
                        weights['lgbm'] * pred_lgbm_proba + 
                        weights['rf'] * pred_rf_proba)

    # --- 5. Create Submission File ---
    print("5. Creating submission file...")
    submission_df = pd.DataFrame({'id': test_ids, 'y': final_pred_proba})
    
    submission_path = "submission.csv"
    submission_df.to_csv(submission_path, index=False)

    print(f"\nWeighted ensemble submission file created at '{submission_path}'")
    print(submission_df.head())


if __name__ == '__main__':
    predict_weighted_ensemble()
