import os
import yaml
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def preprocess_data(config_path):
    """
    Loads the Bank Marketing dataset, performs feature engineering, encoding, 
    scaling, and saves the processed dataframes based on user's analysis.
    """
    # Load configuration from params.yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # --- 1. Load Data & Define Paths ---
    print("1. Loading raw data...")
    raw_dir = config['data_source']['raw_dir']
    train_raw_path = os.path.join(raw_dir, config['data_source']['train_csv'])
    
    processed_dir = config['processed_data']['dir']
    train_processed_path = os.path.join(processed_dir, config['processed_data']['train_csv'])
    
    df_raw = pd.read_csv(train_raw_path)

    # Separate target variable
    target_col = 'y'
    X = df_raw.drop(columns=[target_col])
    y = df_raw[[target_col]]

    # --- 2. Feature Engineering ---
    print("2. Performing feature engineering...")
    X['contacted_previously'] = (X['pdays'] != -1).astype(int)
    print("   - Created 'contacted_previously' feature.")

    # --- 3. Define Feature Groups ---
    numerical_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    nominal_features = ['marital', 'contact', 'month', 'poutcome', 'job'] 
    binary_features = ['default', 'housing', 'loan']
    ordinal_features = ['education'] # This has a specific order

    # --- 4. Create Preprocessing Pipelines ---
    print("4. Building preprocessing pipelines...")

    # Pipeline for numerical features: Scale them.
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # Pipeline for nominal features: Impute 'unknown', then one-hot encode.
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(missing_values='unknown', strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])

    # Create the master preprocessor object using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, ['job', 'marital', 'contact', 'month', 'poutcome']),
            ('bin', OrdinalEncoder(categories=[['no', 'yes']] * len(binary_features)), binary_features),
            ('ord', Pipeline(steps=[
                ('imputer', SimpleImputer(missing_values='unknown', strategy='most_frequent')),
                ('ordinal', OrdinalEncoder(categories=[['primary', 'secondary', 'tertiary']]))
            ]), ['education'])
        ],
        remainder='passthrough' # This will keep our new 'contacted_previously' feature
    )

    # --- 5. Apply Transformations ---
    print("5. Applying transformations to the data...")
    X_processed = preprocessor.fit_transform(X)

    # --- 6. Recreate DataFrame & Save ---
    print("6. Recreating processed DataFrame and saving...")
    
    # FIX: Use the preprocessor's get_feature_names_out() method.
    # This is the modern, robust way to get all column names in the correct order.
    new_cols = preprocessor.get_feature_names_out()

    X_processed = pd.DataFrame(X_processed, columns=new_cols)
    
    # Map the target column 'y' to 1/0
    y_processed = y.replace({'yes': 1, 'no': 0})

    # Combine processed features and target
    df_processed = pd.concat([X_processed, y_processed.reset_index(drop=True)], axis=1)

    # Ensure the output directory exists
    os.makedirs(processed_dir, exist_ok=True)
    
    df_processed.to_csv(train_processed_path, index=False)
    
    print("\nPreprocessing complete.")
    print(f"Processed data shape: {df_processed.shape}")
    print(f"Saved processed data to '{train_processed_path}'")

if __name__ == '__main__':
    preprocess_data(config_path='params.yaml')
