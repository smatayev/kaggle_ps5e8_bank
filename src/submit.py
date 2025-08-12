import hydra
from omegaconf import DictConfig
import subprocess
import os

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def submit_to_kaggle(cfg: DictConfig):
    """
    Submits the prediction file to the specified Kaggle competition
    using the Kaggle API.
    """
    competition = cfg.kaggle.competition_name
    submission_file = cfg.kaggle.submission_file
    message = cfg.kaggle.submission_message

    # Check if the submission file exists
    if not os.path.exists(submission_file):
        print(f"Error: Submission file not found at '{submission_file}'")
        print("Please run the prediction script first.")
        return

    print(f"Submitting '{submission_file}' to the '{competition}' competition...")

    # Construct the command-line command
    command = [
        "kaggle",
        "competitions",
        "submit",
        "-c", competition,
        "-f", submission_file,
        "-m", message
    ]

    # Execute the command
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("Submission successful!")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Submission failed.")
        print("Error details:")
        print(e.stderr)

if __name__ == "__main__":
    submit_to_kaggle()
