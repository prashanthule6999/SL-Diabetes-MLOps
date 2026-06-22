# register model

from src.data.data_ingestion import load_params
import json
import time
import os
import mlflow
# import dagshub
import warnings
from src.logger import logging
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore", UserWarning)


def setup_mlflow(config):

    mode = config["mlflow"]["tracking_mode"].lower()

    if mode == "local":
        mlflow.set_tracking_uri("file:./mlruns")
    elif mode == "dagshub":

        token = os.getenv("SL_DIABETES")

        if not token:
            raise EnvironmentError(
                "SL_DIABETES environment variable is not set"
            )

        os.environ["MLFLOW_TRACKING_USERNAME"] = token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token

        mlflow.set_tracking_uri(
            config["mlflow"]["tracking_uri"]
        )

    else:
        raise ValueError(
            f"Invalid tracking mode: {mode}"
        )

    mlflow.set_experiment(
        config["mlflow"]["experiment_name"]
    )

    logging.info("Tracking Mode: %s", mode)
    logging.info("Tracking URI: %s", mlflow.get_tracking_uri())


def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logging.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error(
            'Unexpected error occurred while loading the model info: %s', e)
        raise


def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry."""
    try:
        # model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        model_uri = model_info["model_uri"]

        # Register the model
        model_version = mlflow.register_model(model_uri, model_name)

        # Assign staging alias to the registered model
        client = mlflow.tracking.MlflowClient()

        ready = False
        # wait for model creation
        for _ in range(30):
            mv = client.get_model_version(
                name=model_name,
                version=model_version.version
            )

            if mv.status == "READY":
                logging.info("Model version is READY")
                ready = True
                break

            time.sleep(3)

        if not ready:
            raise TimeoutError("Model version did not become READY in time")

        client.set_registered_model_alias(
            name=model_name,
            alias="staging",
            version=model_version.version
        )

        logging.info(
            "Assigned alias 'staging' to version %s",
            model_version.version
        )

        logging.info(
            "Model '%s' version %s registered successfully",
            model_name,
            model_version.version
        )

        registration_info = {
            "model_name": model_name,
            "version": model_version.version,
            "model_uri": model_uri,
            "alias": "staging"
        }

        with open(
            "reports/registration_info.json",
            "w"
        ) as f:
            json.dump(registration_info, f, indent=4)
    except Exception as e:
        logging.error('Error during model registration: %s', e)
        raise


def main():
    try:
        params = load_params("params.yaml")
        setup_mlflow(params)

        model_info_path = 'reports/experiment_info.json'
        model_info = load_model_info(model_info_path)

        model_name = params["mlflow"]["model_name"]
        register_model(model_name, model_info)
    except Exception as e:
        logging.error(
            'Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
