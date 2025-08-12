import os
import yaml
import pandas as pd
import joblib

def predict(config_path):
    """
    Loads the trained model and preprocessor, processes the raw test data,
    and generates a submission file.
    """
    # Load configuration from params.yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # --- 1. Load Artifacts and Raw Test Data ---
    print("1. Loading artifacts and raw test data...")
    model_dir = config['model']['dir']
    model_path = os.path.join(model_dir, "final_model.joblib")
    
    # Correctly load the preprocessor from its dedicated directory
    preprocessor_dir = config['preprocessor']['dir']
    preprocessor_path = os.path.join(preprocessor_dir, config['preprocessor']['filename'])
    
    # Load the raw test data that corresponds to the Kaggle submission
    raw_dir = config['data_source']['raw_dir']
    test_raw_path = os.path.join(raw_dir, config['data_source']['test_csv'])

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    df_test_raw = pd.read_csv(test_raw_path)

    # Keep the 'id' column for the submission file
    if 'id' in df_test_raw.columns:
        test_ids = df_test_raw['id']
    else:
        test_ids = pd.Series(range(len(df_test_raw)), name="id")


    # --- 2. Preprocess the Test Data ---
    print("2. Preprocessing the test data...")
    
    # FIX: Perform the same feature engineering on the test set
    # This creates the 'contacted_previously' column that the preprocessor expects.
    print("   - Creating 'contacted_previously' feature for test set.")
    df_test_raw['contacted_previously'] = (df_test_raw['pdays'] != -1).astype(int)

    # Apply the SAME transformations that were fitted on the training data
    X_test_processed = preprocessor.transform(df_test_raw)
    
    # --- 3. Generate Predictions ---
    print("3. Generating predictions...")
    predictions = model.predict(X_test_processed)

    # --- 4. Create Submission File ---
    print("4. Creating submission file...")
    submission_df = pd.DataFrame({'id': test_ids, 'y': predictions})
    
    # Map the numerical predictions back to 'yes'/'no'
    #submission_df['y'] = submission_df['y'].replace({1: 'yes', 0: 'no'})
    
    submission_path = "submission.csv"
    submission_df.to_csv(submission_path, index=False)

    print(f"\nSubmission file created at '{submission_path}'")
    print(submission_df.head())


if __name__ == '__main__':
    predict(config_path='params.yaml')
