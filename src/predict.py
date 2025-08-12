import os
import hydra
from omegaconf import DictConfig
import pandas as pd
import joblib

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def predict(cfg: DictConfig):
    """
    Loads the FINAL trained model and preprocessor, processes the raw test data,
    and generates a submission file using configuration from Hydra.
    """
    # --- 1. Load Artifacts and Raw Test Data ---
    print("1. Loading artifacts and raw test data...")
    model_dir = hydra.utils.to_absolute_path(cfg.model.dir)
    final_model_filename = cfg.model.final_filename
    model_path = os.path.join(model_dir, final_model_filename)
    
    preprocessor_dir = hydra.utils.to_absolute_path(cfg.preprocessor.dir)
    preprocessor_path = os.path.join(preprocessor_dir, cfg.preprocessor.filename)
    
    raw_dir = hydra.utils.to_absolute_path(cfg.data_source.raw_dir)
    test_raw_path = os.path.join(raw_dir, cfg.data_source.test_csv)

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    df_test_raw = pd.read_csv(test_raw_path)

    if 'id' in df_test_raw.columns:
        test_ids = df_test_raw['id']
    else:
        test_ids = pd.Series(range(len(df_test_raw)), name="id")

    # --- 2. Preprocess the Test Data ---
    print("2. Preprocessing the test data...")
    
    print("   - Creating 'contacted_previously' feature for test set.")
    df_test_raw['contacted_previously'] = (df_test_raw['pdays'] != -1).astype(int)

    X_test_processed = preprocessor.transform(df_test_raw)
    
    # --- 3. Generate Predictions ---
    print("3. Generating predictions...")
    predictions = model.predict(X_test_processed)

    # --- 4. Create Submission File ---
    print("4. Creating submission file...")
    submission_df = pd.DataFrame({'id': test_ids, 'y': predictions})
    
    # The submission file is saved in the root of the project directory
    submission_path = "submission.csv"
    submission_df.to_csv(submission_path, index=False)

    print(f"\nSubmission file created at '{submission_path}'")
    print(submission_df.head())


if __name__ == '__main__':
    predict()
