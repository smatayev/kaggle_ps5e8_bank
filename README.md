# End-to-End MLOps: Bank Marketing Classification

## 1. Project Overview

This project demonstrates a complete, end-to-end Machine Learning Operations (MLOps) workflow for a classic classification problem using the Bank Marketing dataset. The goal is to predict whether a client will subscribe to a term deposit.

This repository serves as a portfolio piece to showcase best practices in building reproducible, version-controlled, and automated machine learning pipelines.

---

## 2. MLOps Concepts & Practices

This project was built from the ground up to incorporate the following MLOps principles:

* **💻 Modular Code:** The project is organized into distinct, single-responsibility scripts (`preprocess.py`, `split.py`, `train_and_evaluate.py`) located in the `src/` directory.
* **📦 Environment Management:** A dedicated Python virtual environment and a `requirements.txt` file ensure that the project's dependencies are locked and can be easily recreated.
* **Reproducibility**: The combination of Git, DVC, and a locked `requirements.txt` file ensures that any experiment can be perfectly reproduced.
* **Automation**: The entire pipeline, from data preprocessing to model evaluation, is automated with `dvc repro`. The final submission process is automated with GitHub Actions.
* **Modularity**: The code is organized into distinct scripts (`preprocess`, `train`, `optimize`, etc.), and the configuration is broken down into modular files, making the system easy to extend and maintain.


---

## Technology Stack & Purpose

This project leverages a modern MLOps stack to streamline experimentation and ensure reproducibility.

| Tool | Purpose |
| :--- | :--- |
| **Git & GitHub** | For versioning all code, configuration, and pipeline definitions. |
| **DVC** | For versioning large data files and model artifacts, keeping the Git repo lightweight. |
| **Hydra** | For managing complex configurations, allowing for easy experimentation by swapping models or preprocessing steps from the command line. |
| **MLflow** | For logging and tracking all experiment runs, including parameters, metrics, and model artifacts, with a UI for easy comparison. |
| **Optuna** | For automated hyperparameter optimization to find the best-performing models efficiently. |
| **GitHub Actions**| For CI/CD, creating a fully automated workflow that trains, predicts, and submits to Kaggle on every push to the `main` branch. |
| **Python** | The core programming language. |
| **Pandas & Scikit-learn** | For data manipulation and building preprocessing pipelines. |
| **XGBoost & LightGBM** | High-performance gradient-boosting libraries for modeling. |

---

## 4. How to Reproduce the Project

To set up and run this project on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [your-repo-url]
    cd [repo-name]
    ```
2.  **Set up the environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
3.  **Pull the data:**
    ```bash
    dvc pull
    ```
4.  **Run the evaluation pipeline:**
    ```bash
    dvc repro
    ```
5.  **View experiments:**
    ```bash
    mlflow ui
    ```