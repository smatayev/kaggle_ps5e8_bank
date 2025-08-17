# MLOps Project: An End-to-End Journey in Competitive Data Science

## 1. Project Overview

This repository documents a complete MLOps workflow built to tackle the [Bank Marketing Kaggle Competition](https://www.kaggle.com/competitions/playground-series-s5-e8). The primary goal was not just to build a single predictive model, but to create a robust, automated, and reproducible experimentation platform designed to systematically improve a baseline score and climb the leaderboard.

This project serves as a practical demonstration of modern MLOps principles, from initial setup to advanced ensembling and automated submission.

---

## 2. The MLOps Workflow & Technology Stack

The project was built on a foundation of modularity, automation, and version control, using a modern MLOps toolchain.

| Tool | Purpose |
| :--- | :--- |
| **Git & GitHub** | For versioning all code, configuration, and pipeline definitions. |
| **DVC** | Initially used for data versioning, later replaced with direct Kaggle API ingestion for a simpler CI/CD workflow. |
| **Hydra** | For managing a modular and dynamic configuration system, allowing for easy experimentation by swapping models or preprocessing steps from the command line. |
| **MLflow** | For logging and tracking all experiment runs, providing a UI to compare parameters, metrics, and model artifacts. |
| **Optuna** | For automated and intelligent hyperparameter optimization. |
| **GitHub Actions**| For a fully automated CI/CD pipeline that ingests data, trains the final model, generates predictions, and submits to Kaggle on every push to the `main` branch. |
| **Python** | The core programming language. |
| **Scikit-learn, XGBoost, LightGBM** | For data preprocessing and building a diverse set of high-performance models. |

---

## 3. The Experimentation Journey

The project followed a systematic process of iterative improvement, with each step logged and evaluated.

#### **Baseline (Score: ~0.81)**
* Established a solid project structure with a virtual environment, Git for code versioning, and an initial DVC pipeline for data versioning.
* Trained a baseline **XGBoost** model with a simple preprocessing pipeline.

#### **Optimization & Refactoring (Score: 0.9669)**
* Refactored the entire project to use **Hydra** for a flexible configuration system.
* Integrated **MLflow** to automatically track all experiments.
* Used **Optuna** to perform automated hyperparameter tuning on the XGBoost model, resulting in a significant score increase.

#### **Adding Model Diversity (Score: 0.9660)**
* Integrated **LightGBM** and **RandomForestClassifier** into the workflow as new, diverse models.
* Used Optuna to tune these new models, establishing strong, independent baselines for each.

#### **Advanced Feature Engineering**
* **Aggregate Features**: Implemented a preprocessor to create statistical features (e.g., mean/std of `balance` per `job` type). This did not yield a score improvement, suggesting the tree-based models were already capturing these interactions.
* **Target Encoding**: Implemented a safe, cross-validated target encoding strategy. This also did not provide a lift over the baseline, providing valuable evidence that the current feature set was well-optimized.

#### **Ensembling for the Final Push (Top Score: 0.96808)**
* **Simple Averaging**: An initial ensemble that averaged the predictions of the two best models (XGBoost and LightGBM) provided a strong score boost.
* **Weighted Averaging**: Experimented with different weights to favor the better-performing models.
* **Stacking**: Implemented a full stacking ensemble with a `LogisticRegression` meta-model to learn the optimal way to combine the base models. This was further improved by tuning the meta-model itself with Optuna, resulting in the project's highest score.
* **Final Blending**: The final submissions were created by blending the predictions of the best stacking and weighted-average models.

---

## 4. How to Reproduce the Project

1.  **Clone the repository and set up the environment:**
    ```bash
    git clone [your-repo-url]
    cd [repo-name]
    python3 -m venv venv && source venv/bin/activate
    pip install -r requirements.txt
    ```
2.  **Set up Kaggle API credentials.**

3.  **Run a single experiment:**
    (e.g., train the optimized LightGBM model)
    ```bash
    python src/train_and_evaluate.py model=lightgbm
    ```
4.  **View all experiments:**
    ```bash
    mlflow ui
    ```
5.  **Generate the final stacking submission:**
    ```bash
    python src/stacking.py
    ```
