# End-to-End MLOps: Bank Marketing Classification

## 1. Project Overview

This project demonstrates a complete, end-to-end Machine Learning Operations (MLOps) workflow for a classic classification problem using the Bank Marketing dataset. The goal is to predict whether a client will subscribe to a term deposit.

This repository serves as a portfolio piece to showcase best practices in building reproducible, version-controlled, and automated machine learning pipelines.

---

## 2. MLOps Concepts & Practices

This project was built from the ground up to incorporate the following MLOps principles:

* **💻 Modular Code:** The project is organized into distinct, single-responsibility scripts (`preprocess.py`, `split.py`, `train_and_evaluate.py`) located in the `src/` directory.
* **📜 Centralized Configuration:** All pipeline parameters (file paths, model hyperparameters, random states) are managed in a single `params.yaml` file, separating configuration from code.
* **🔍 Version Control:**
    * **Git & GitHub:** Used for versioning all code and configuration files.
    * **DVC (Data Version Control):** Used to version large data files and model artifacts without bloating the Git repository. This ensures that every experiment is fully reproducible, from the exact data version to the final model.
* **🔁 Reproducible Pipelines:** The entire workflow, from data preprocessing to model training and evaluation, is defined in `dvc.yaml`. The pipeline can be reproduced with a single command (`dvc repro`), ensuring consistency and saving time by only re-running stages affected by changes.
* **📦 Environment Management:** A dedicated Python virtual environment and a `requirements.txt` file ensure that the project's dependencies are locked and can be easily recreated.
* **🧪 Experiment Management:** The structure allows for easy experimentation by modifying `params.yaml` and using DVC to track and compare the resulting metrics.

---

## 3. Technology Stack

* **Languages:** Python 3.9
* **Core Libraries:** Pandas, Scikit-learn, XGBoost, Imbalanced-learn
* **MLOps Tools:** Git, DVC
* **Environment:** venv, Homebrew (for `libomp`)

---

## 4. How to Reproduce the Project

To set up and run this project on your local machine, follow these steps:

1.  **Clone the Repository:**
    ```bash
    git clone [https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-new-repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-new-repository)
    cd [your-repo-name]
    ```

2.  **Create and Activate Virtual Environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    * **(macOS Only) Install OpenMP:**
        ```bash
        brew install libomp
        ```
    * **Install Python packages:**
        ```bash
        pip install -r requirements.txt
        ```

4.  **Pull Data from DVC Remote:** (n/a, localremote used. you can download the raw file from Kaggle and reproduce preprocessing steps)
    ```bash
    dvc pull
    ```
    This command will download the version-controlled data and model artifacts into your workspace.

5.  **Reproduce the Pipeline:**
    ```bash
    dvc repro
    ```
    This single command will execute the entire pipeline (`preprocess` -> `split` -> `train_and_evaluate`) and generate the final model and metrics.

---

## 5. Pipeline Workflow

The project is automated via the `dvc.yaml` file, which defines the following three stages:

1.  **`preprocess`**:
    * **Inputs:** Raw data from `data/raw/`.
    * **Action:** Cleans the data, performs feature engineering, scaling, and encoding.
    * **Outputs:** Processed data in `data/processed/` and a fitted preprocessor object in `artifacts/preprocessor/`.

2.  **`split`**:
    * **Inputs:** Processed data from `data/processed/`.
    * **Action:** Splits the data into training and validation sets.
    * **Outputs:** Data splits in `data/split/`.

3.  **`train_and_evaluate`**:
    * **Inputs:** Data splits from `data/split/` and model parameters from `params.yaml`.
    * **Action:** Handles class imbalance using SMOTE, trains an XGBoost model, and evaluates its performance.
    * **Outputs:** A trained model in `artifacts/model/` and performance metrics in `artifacts/metrics/`.

---

## 6. Final Steps

To generate a `submission.csv` file for Kaggle using the final model trained on the full dataset, run the following scripts in order:

```bash
# Train the model on 100% of the data
python src/train_final_model.py

# Generate predictions on the test set
python src/predict.py
