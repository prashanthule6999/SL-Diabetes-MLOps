#  model building

import os
import pickle
import warnings
import pandas as pd
from typing import Optional
from src.logger import logging
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.data.data_ingestion import load_params
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error(
            'Unexpected error occurred while loading the data: %s', e)
        raise


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

        logging.info("NaNs in X_train: %s", X_train.isna().sum().sum())
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


def save_model(model: Pipeline, file_path: str) -> None:
    """Save the trained model to a file."""
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logging.info('Model saved to %s', file_path)
        # logging.info("Model parameters: %s", model.get_params())
    except Exception as e:
        logging.error('Error occurred while saving the model: %s', e)
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

        train_data = load_data('./data/interim/train_processed.csv')

        # X_train = train_data.iloc[:, :-1]
        # y_train = train_data.iloc[:, -1]
        X_train = train_data.drop(columns=[target_column])
        y_train = train_data[target_column]

        clf = train_model(X_train, y_train, penalty, solver,
                          C, class_weight, random_state)

        os.makedirs("models", exist_ok=True)
        save_model(clf, 'models/model.pkl')

    except Exception as e:
        logging.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
