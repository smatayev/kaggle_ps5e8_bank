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
from utils import reduce_mem_usage

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def preprocess_data(cfg: DictConfig):
    """
    Loads data, reduces memory, and applies preprocessing and feature
    engineering based on the Hydra configuration.
    """
    # --- 1. Load & Prep Data ---
    print("1. Loading and preparing data...")
    raw_dir = hydra.utils.to_absolute_path(cfg.data_source.raw_dir)
    train_raw_path = os.path.join(raw_dir, cfg.data_source.train_csv)
    df_raw = pd.read_csv(train_raw_path)
    df_raw = reduce_mem_usage(df_raw)

    target_col = cfg.base.target_col
    X = df_raw.drop(columns=[target_col])
    y = df_raw[[target_col]]
    X['contacted_previously'] = (X['pdays'] != -1).astype(int)

    # --- 2. Advanced Feature Engineering (Conditional) ---
    if cfg.preprocessor.name == "v3_aggregates":
        print("\n2. Applying aggregate feature engineering (v3)...")
        
        for spec in cfg.preprocessor.aggregate_features:
            group_col = spec.group_by
            for agg_col in spec.agg_cols:
                # Define aggregations
                aggregations = {'mean', 'std', 'max', 'min'}
                for agg_type in aggregations:
                    new_col_name = f"{group_col}_{agg_col}_{agg_type}"
                    print(f"   - Creating '{new_col_name}'")
                    
                    # Calculate the aggregate
                    agg_value = X.groupby(group_col)[agg_col].transform(agg_type)
                    
                    # Create the new feature
                    X[new_col_name] = agg_value

    # --- 3. Define Feature Groups for Preprocessing ---
    print("\n3. Defining feature groups for preprocessing...")
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features_for_ohe = ['marital', 'contact', 'month', 'poutcome', 'job'] 
    binary_features = ['default', 'housing', 'loan']
    ordinal_features = ['education']
    
    for col in categorical_features_for_ohe + binary_features + ordinal_features:
        if col in numerical_features:
            numerical_features.remove(col)

    # --- 4. Create and Apply Preprocessing Pipeline ---
    print("\n4. Building and applying preprocessing pipeline...")
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(missing_values='unknown', strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features_for_ohe),
            ('bin', OrdinalEncoder(categories=[['no', 'yes']] * len(binary_features)), binary_features),
            ('ord', Pipeline(steps=[
                ('imputer', SimpleImputer(missing_values='unknown', strategy='most_frequent')),
                ('ordinal', OrdinalEncoder(categories=[['primary', 'secondary', 'tertiary']]))
            ]), ['education'])
        ],
        remainder='drop'
    )

    X_processed = preprocessor.fit_transform(X)

    # --- 5. Save Processed Data and Artifacts ---
    print("\n5. Saving processed data and artifacts...")
    new_cols = preprocessor.get_feature_names_out()
    X_processed = pd.DataFrame(X_processed, columns=new_cols)
    y_processed = y.replace({'yes': 1, 'no': 0})
    df_processed = pd.concat([X_processed, y_processed.reset_index(drop=True)], axis=1)
    
    processed_dir = hydra.utils.to_absolute_path(cfg.processed_data.dir)
    os.makedirs(processed_dir, exist_ok=True)
    df_processed.to_csv(os.path.join(processed_dir, cfg.processed_data.train_csv), index=False)
    
    preprocessor_dir = hydra.utils.to_absolute_path(cfg.preprocessor.dir)
    os.makedirs(preprocessor_dir, exist_ok=True)
    joblib.dump(preprocessor, os.path.join(preprocessor_dir, cfg.preprocessor.filename))

    print("\nPreprocessing complete.")

if __name__ == '__main__':
    preprocess_data()
