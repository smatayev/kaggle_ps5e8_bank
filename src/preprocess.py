import os
import hydra
from omegaconf import DictConfig
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
from utils import reduce_mem_usage # Import the custom memory utils function

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def preprocess_data(cfg: DictConfig):
    """
    Loads the raw dataset, reduces its memory usage, performs 
    feature engineering, encoding, scaling, and saves the processed dataframes 
    and the preprocessor object.
    """
    # --- 1. Load Data & Define Paths ---
    print("1. Loading raw data...")
    # Hydra automatically changes the working directory to the output folder.
    # We use hydra.utils.to_absolute_path to get the correct path to our data.
    raw_dir = hydra.utils.to_absolute_path(cfg.data_source.raw_dir)
    train_raw_path = os.path.join(raw_dir, cfg.data_source.train_csv)
    
    df_raw = pd.read_csv(train_raw_path)

    # --- 2. Reduce Memory Usage ---
    print("\n2. Optimizing memory usage...")
    df_raw = reduce_mem_usage(df_raw) # Apply the utility function

    # Separate target variable
    target_col = cfg.base.target_col
    X = df_raw.drop(columns=[target_col])
    y = df_raw[[target_col]]

    # --- 3. Feature Engineering ---
    print("\n3. Performing feature engineering...")
    X['contacted_previously'] = (X['pdays'] != -1).astype(int)
    print("   - Created 'contacted_previously' feature.")

    # --- 4. Define Feature Groups ---
    numerical_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    nominal_features = ['marital', 'contact', 'month', 'poutcome', 'job'] 
    binary_features = ['default', 'housing', 'loan']
    ordinal_features = ['education']

    # --- 5. Create Preprocessing Pipelines ---
    print("\n5. Building preprocessing pipelines...")
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(missing_values='unknown', strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])

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
        remainder='passthrough'
    )

    # --- 6. Apply Transformations ---
    print("\n6. Applying transformations to the data...")
    X_processed = preprocessor.fit_transform(X)

    # --- 7. Recreate DataFrame & Save Data ---
    print("\n7. Recreating processed DataFrame and saving...")
    new_cols = preprocessor.get_feature_names_out()
    X_processed = pd.DataFrame(X_processed, columns=new_cols)
    y_processed = y.replace({'yes': 1, 'no': 0})
    df_processed = pd.concat([X_processed, y_processed.reset_index(drop=True)], axis=1)
    
    # Use absolute paths for saving outputs as well
    processed_dir = hydra.utils.to_absolute_path(cfg.processed_data.dir)
    os.makedirs(processed_dir, exist_ok=True)
    processed_train_path = os.path.join(processed_dir, cfg.processed_data.train_csv)
    df_processed.to_csv(processed_train_path, index=False)
    
    # --- 8. Save the Preprocessor Object ---
    preprocessor_dir = hydra.utils.to_absolute_path(cfg.preprocessor.dir)
    preprocessor_path = os.path.join(preprocessor_dir, cfg.preprocessor.filename)
    os.makedirs(preprocessor_dir, exist_ok=True)
    print(f"8. Saving preprocessor to '{preprocessor_path}'...")
    joblib.dump(preprocessor, preprocessor_path)

    print("\nPreprocessing complete.")

if __name__ == '__main__':
    preprocess_data()
