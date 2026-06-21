
import os
import json
import mlflow
import pickle
import dagshub
import pandas as pd
import mlflow.sklearn
from typing import Dict, Any
from src.logger import logging
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from src.data.data_ingestion import load_params
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score


def setup_mlflow(config):

    mode = config["mlflow"]["tracking_mode"]

    if mode == "local":
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
    elif mode == "dagshub":

        token = os.getenv("SL_DIABETES")

        if not token:
            raise EnvironmentError(
                "SL_DIABETES environment variable is not set"
            )

        os.environ["MLFLOW_TRACKING_USERNAME"] = token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token

        dagshub.init(
            repo_owner=config["mlflow"]["repo_owner"],
            repo_name=config["mlflow"]["repo_name"],
            mlflow=True
        )

    mlflow.set_experiment(
        config["mlflow"]["experiment_name"]
    )


# def load_model(file_path: str):
#     """Load the trained model from a file."""
#     try:
#         with open(file_path, 'rb') as file:
#             model = pickle.load(file)
#         logging.info('Model loaded from %s', file_path)
#         return model
#     except FileNotFoundError:
#         logging.error('File not found: %s', file_path)
#         raise
#     except Exception as e:
#         logging.error(
#             'Unexpected error occurred while loading the model: %s', e)
#         raise

def load_artifacts(file_path: str) -> Dict[str, Any]:
    """Load the artifacts(model, threshold) from a file."""
    try:
        with open(file_path, "rb") as f:
            artifacts = pickle.load(f)

        logging.info('Artifacts loaded from %s', file_path)
        return artifacts
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error(
            'Unexpected error occurred while loading the artifacts: %s', e)
        raise


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


def evaluate_model(clf: Pipeline, best_threshold: float, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Evaluate the model and return the evaluation metrics."""
    try:
        test_prob = clf.predict_proba(X_test)[:, 1]
        test_pred = (test_prob >= best_threshold).astype(int)

        metrics_dict = {
            'test_accuracy': accuracy_score(y_test, test_pred),
            'precision': precision_score(y_test, test_pred),
            'recall': recall_score(y_test, test_pred),
            'f1': f1_score(y_test, test_pred),
            'auc': roc_auc_score(y_test, test_prob)
        }

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            test_pred
        ).ravel()

        metrics_dict.update({
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        })

        logging.info('Model evaluation metrics calculated')
        return metrics_dict
    except Exception as e:
        logging.error('Error during model evaluation: %s', e)
        raise


def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logging.info('Metrics saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the metrics: %s', e)
        raise


def save_model_info(run_id: str, model_uri: str, file_path: str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        model_info = {"run_id": run_id,
                      "model_uri": model_uri}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logging.debug('Model info saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the model info: %s', e)
        raise


def main():
    params = load_params("params.yaml")

    setup_mlflow(params)

    print("Tracking URI:", mlflow.get_tracking_uri())
    with mlflow.start_run() as run:  # Start an MLflow run
        try:
            logging.info("MLflow Run ID: %s", run.info.run_id)
            # load saved pipeline/model & threshold
            artifacts = load_artifacts("models/model_artifact.pkl")
            clf = artifacts["model"]
            best_threshold = artifacts["threshold"]

            target_column = params["model_evaluation"]['target_column']

            test_data = load_data('./data/interim/test_processed.csv')
            X_test = test_data.drop(columns=[target_column])
            y_test = test_data[target_column]

            metrics = evaluate_model(clf, best_threshold, X_test, y_test)

            os.makedirs("reports", exist_ok=True)
            metrics["decision_threshold"] = best_threshold
            save_metrics(metrics, 'reports/metrics.json')

            # Log metrics to MLflow
            # for metric_name, metric_value in metrics.items():
            #     mlflow.log_metric(metric_name, metric_value)
            mlflow.log_metrics(metrics)

            # Log model parameters to MLflow
            if hasattr(clf, 'get_params'):
                # params = clf.get_params()
                # for param_name, param_value in params.items():
                #     mlflow.log_param(param_name, param_value)
                # mlflow.log_params(clf.get_params())
                mlflow.log_param("decision_threshold", best_threshold)
                mlflow.log_params(
                    clf.named_steps["model"].get_params()
                )

            # Log model to MLflow
            model_info = mlflow.sklearn.log_model(clf,
                                                  artifact_path="model")

            # Save model info
            save_model_info(run.info.run_id, model_info.model_uri,
                            'reports/experiment_info.json')

            # Log the artifact/metrics file to MLflow
            mlflow.log_artifact('reports/metrics.json')
            mlflow.log_artifact("reports/experiment_info.json")
            mlflow.log_artifact(
                "models/model_artifact.pkl",
                artifact_path="inference_artifacts"
            )

        except Exception as e:
            logging.error(
                'Failed to complete the model evaluation process: %s', e)
            print(f"Error: {e}")


if __name__ == '__main__':
    main()
