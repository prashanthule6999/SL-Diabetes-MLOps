#  model building

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from typing import Optional
from src.logger import logging
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from src.helper_func.utility import load_params, load_data_from_csv
warnings.filterwarnings("ignore")


def train_model(X_train: pd.DataFrame, y_train: pd.Series, penalty: str, solver: str, C: float, class_weight: Optional[str], random_state: int) -> Pipeline:
    """Train the Logistic Regression model."""
    try:
        model = LogisticRegression(
            penalty=penalty,
            solver=solver,
            C=C,
            class_weight=class_weight,
            random_state=random_state  # for reproducibility
        )

        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model)
        ])

        if y_train.isnull().sum() > 0:
            raise ValueError("Target variable contains missing values")

        logging.info("NaNs in X_train before imputation: %s",
                     X_train.isna().sum().sum())
        logging.info("NaNs in y_train: %s", y_train.isna().sum())

        pipeline.fit(X_train, y_train)
        logging.info(
            "Model converged in %s iterations",
            pipeline.named_steps["model"].n_iter_
        )
        logging.info('Model training completed')

        return pipeline
    except Exception as e:
        logging.error('Error during model training: %s', e)
        raise


def get_best_threshold(clf: Pipeline, X_val: pd.DataFrame, y_val: pd.Series) -> float:

    # using Youden's J statistic for threshold selection
    val_prob = clf.predict_proba(X_val)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_val, val_prob)

    best_idx = np.argmax(tpr - fpr)
    best_threshold = thresholds[best_idx]

    logging.info(
        "Best threshold %.4f selected using Youden's J statistic",
        best_threshold
    )

    return best_threshold


def save_artifacts(clf: Pipeline,  best_threshold: float, file_path: str) -> None:
    """Save the trained model & best threshold to a file."""
    try:
        artifacts = {
            "model": clf,
            "threshold": best_threshold
        }

        with open(file_path, "wb") as f:
            pickle.dump(artifacts, f)

        logging.info('Artifacts saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the artifacts: %s', e)
        raise


def main():
    try:

        params = load_params('params.yaml')
        target_column = params["model_building"]['target_column']
        penalty = params["model_building"]['penalty']
        solver = params["model_building"]['solver']
        C = params["model_building"]['C']
        class_weight = params["model_building"]['class_weight']
        random_state = params["model_building"]['random_state']

        # model training
        train_data = load_data_from_csv('./data/interim/train_processed.csv')

        # X_train = train_data.iloc[:, :-1]
        # y_train = train_data.iloc[:, -1]
        X_train = train_data.drop(columns=[target_column])
        y_train = train_data[target_column]

        temp_clf = train_model(X_train, y_train, penalty, solver,
                               C, class_weight, random_state)

        # finding & saving best threshold using validation set
        val_data = load_data_from_csv('./data/interim/val_processed.csv')
        X_val = val_data.drop(columns=[target_column])
        y_val = val_data[target_column]

        best_threshold = get_best_threshold(temp_clf, X_val, y_val)

        # Combine train + validation for retraining
        final_train_data = pd.concat(
            [train_data, val_data],
            ignore_index=True
        )

        # Features and target
        X_train = final_train_data.drop(columns=[target_column])
        y_train = final_train_data[target_column]

        final_clf = train_model(X_train, y_train, penalty, solver,
                                C, class_weight, random_state)

        # saving both model as well as best threshold
        os.makedirs("models", exist_ok=True)

        # The principle is :
        # Train once → Save all learned transformations + model → Load the saved artifact for evaluation/inference.
        # That's a production-friendly pattern because evaluation and deployment use the exact same trained artifact
        # rather than rebuilding preprocessing from code each time.

        # Final evaluation on the untouched test set
        # is performed in model_evaluation.py
        save_artifacts(final_clf, best_threshold, 'models/model_artifact.pkl')

        logging.info(
            "Saved threshold %.4f with trained model",
            best_threshold
        )

    except Exception as e:
        logging.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
