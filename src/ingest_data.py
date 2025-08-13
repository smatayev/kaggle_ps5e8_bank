import hydra
from omegaconf import DictConfig
import subprocess
import os
from zipfile import ZipFile

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def ingest_data(cfg: DictConfig):
    """
    Downloads and unzips the competition data from Kaggle.
    """
    competition = cfg.kaggle.competition_name
    raw_data_dir = hydra.utils.to_absolute_path(cfg.data_source.raw_dir)
    
    # Ensure the raw data directory exists
    os.makedirs(raw_data_dir, exist_ok=True)

    print(f"Downloading data for '{competition}' competition...")

    # Construct the command-line command
    command = [
        "kaggle",
        "competitions",
        "download",
        "-c", competition,
        "-p", raw_data_dir,
        "--force" # Overwrite existing files
    ]

    # Execute the command
    try:
        subprocess.run(command, check=True)
        print("Download successful!")
    except subprocess.CalledProcessError as e:
        print("Download failed.")
        print("Error details:", e)
        return

    # Unzip the downloaded file
    zip_path = os.path.join(raw_data_dir, f"{competition}.zip")
    if os.path.exists(zip_path):
        print(f"Unzipping '{zip_path}'...")
        with ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(raw_data_dir)
        # Remove the zip file after extraction
        os.remove(zip_path)
        print("Unzipping complete.")
    else:
        print(f"Error: Zip file not found at '{zip_path}'")


if __name__ == "__main__":
    ingest_data()
