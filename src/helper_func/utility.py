import os
import yaml
import mlflow
import pandas as pd
from typing import Any
from pathlib import Path
from src.logger import logging

def load_params(params_path: str) -> dict[str, Any]:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logging.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logging.error('YAML error: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error: %s', e)
        raise

def load_data_from_csv(file_path: str) -> pd.DataFrame:
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

def setup_mlflow(config):

    mode = config["mlflow"]["tracking_mode"].lower()

    if mode == "local":
        
        ROOT_DIR = Path(__file__).resolve().parent.parent
        mlflow.set_tracking_uri(f"file:{ROOT_DIR / 'mlruns'}")
        
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
