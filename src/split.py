import os
import hydra
from omegaconf import DictConfig
import pandas as pd
from sklearn.model_selection import train_test_split

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def split_data(cfg: DictConfig):
    """
    Loads the processed data and splits it into training and testing sets
    using configuration from Hydra.
    """
    # --- 1. Define Paths ---
    processed_data_dir = hydra.utils.to_absolute_path(cfg.processed_data.dir)
    processed_train_path = os.path.join(processed_data_dir, cfg.processed_data.train_csv)
    
    split_data_dir = hydra.utils.to_absolute_path(cfg.split_data.dir)
    split_train_path = os.path.join(split_data_dir, cfg.split_data.train_path)
    split_test_path = os.path.join(split_data_dir, cfg.split_data.test_path)

    # --- 2. Load Processed Data ---
    print(f"Loading processed data from '{processed_train_path}'...")
    df = pd.read_csv(processed_train_path)

    # --- 3. Perform the Split ---
    test_size = cfg.split_data.test_size
    random_state = cfg.base.random_state
    target_col = cfg.base.target_col

    print(f"Splitting data with test size: {test_size}...")
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_col]
    )

    # --- 4. Save the Split Data ---
    os.makedirs(split_data_dir, exist_ok=True)
    
    print(f"Saving training split to '{split_train_path}'")
    train_df.to_csv(split_train_path, index=False)
    
    print(f"Saving testing split to '{split_test_path}'")
    test_df.to_csv(split_test_path, index=False)

    print("\nData splitting complete.")
    print(f"Training set shape: {train_df.shape}")
    print(f"Testing set shape: {test_df.shape}")


if __name__ == '__main__':
    split_data()
