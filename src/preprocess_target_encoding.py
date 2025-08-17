import os
import hydra
from omegaconf import DictConfig
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
from utils import reduce_mem_usage

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def preprocess_target_encoding(cfg: DictConfig):
    """
    Performs preprocessing with a focus on safe, cross-validated target encoding
    to prevent data leakage.
    """
    # --- 1. Load & Prep Data ---
    print("1. Loading and preparing data for target encoding...")
    raw_dir = hydra.utils.to_absolute_path(cfg.data_source.raw_dir)
    train_raw_path = os.path.join(raw_dir, cfg.data_source.train_csv)
    df_raw = pd.read_csv(train_raw_path)
    df_raw = reduce_mem_usage(df_raw)

    target_col = cfg.base.target_col
    X = df_raw.drop(columns=[target_col])
    y = df_raw[target_col].replace({'yes': 1, 'no': 0}) # Convert target to numeric for encoding

    X['contacted_previously'] = (X['pdays'] != -1).astype(int)

    # --- 2. Safe Target Encoding with Cross-Validation ---
    print("\n2. Applying cross-validated target encoding...")
    target_encode_cols = cfg.preprocessor.target_encode_cols
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.base.random_state)
    
    # Create a new DataFrame to store the encoded features
    X_encoded = X.copy()

    for col in target_encode_cols:
        print(f"   - Encoding '{col}'")
        # Create a new column to store the out-of-fold encoded values
        X_encoded[f'{col}_te'] = 0.0
        
        for train_idx, val_idx in skf.split(X, y):
            # Calculate the mean of the target for the training part of the fold
            target_mean = y.iloc[train_idx].groupby(X.iloc[train_idx][col]).mean()
            
            # Apply this mean to the validation part of the fold
            X_encoded.iloc[val_idx, X_encoded.columns.get_loc(f'{col}_te')] = X.iloc[val_idx][col].map(target_mean)

        # Fill any potential NaNs with the global mean
        X_encoded[f'{col}_te'].fillna(y.mean(), inplace=True)

    # Drop the original categorical columns that we've encoded
    X_encoded.drop(columns=target_encode_cols, inplace=True)
    
    # --- 3. Standard Preprocessing on Remaining Features ---
    print("\n3. Applying standard preprocessing to other features...")
    numerical_features = X_encoded.select_dtypes(include=np.number).columns.tolist()
    binary_features = ['default', 'housing', 'loan']
    
    # Remove binary features from the numerical list to handle them separately
    for col in binary_features:
        if col in numerical_features:
            numerical_features.remove(col)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('bin', OrdinalEncoder(categories=[['no', 'yes']] * len(binary_features)), binary_features)
        ],
        remainder='passthrough'
    )

    X_processed = preprocessor.fit_transform(X_encoded)

    # --- 4. Save Processed Data and Artifacts ---
    print("\n4. Saving processed data and artifacts...")
    new_cols = preprocessor.get_feature_names_out()
    X_processed = pd.DataFrame(X_processed, columns=new_cols)
    df_processed = pd.concat([X_processed, y.reset_index(drop=True)], axis=1)
    
    processed_dir = hydra.utils.to_absolute_path(cfg.processed_data.dir)
    os.makedirs(processed_dir, exist_ok=True)
    df_processed.to_csv(os.path.join(processed_dir, cfg.processed_data.train_csv), index=False)
    
    preprocessor_dir = hydra.utils.to_absolute_path(cfg.preprocessor.dir)
    os.makedirs(preprocessor_dir, exist_ok=True)
    joblib.dump(preprocessor, os.path.join(preprocessor_dir, cfg.preprocessor.filename))

    print("\nPreprocessing with target encoding complete.")

if __name__ == '__main__':
    preprocess_target_encoding()
