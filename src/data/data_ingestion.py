# data ingestion
from src.logger import logging
import logging
import yaml
from sklearn.model_selection import train_test_split
import os
import kagglehub
import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)


def load_params(params_path: str) -> dict:
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


def load_data(data_url: str, fileName: str) -> pd.DataFrame:
    """Load data from kaggle site."""
    try:
        path = kagglehub.dataset_download(data_url)
        csv_file = os.path.join(path, fileName)
        df = pd.read_csv(csv_file)
        logging.info('Data loaded from %s/%s', data_url, fileName)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error(
            'Unexpected error occurred while loading the data: %s', e)
        raise


def preprocess_data(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Replace zeros with NaN in specified columns."""
    try:
        logging.info("Pre-processing data...")

        final_df = df.copy()
        final_df[cols] = final_df[cols].replace(0, np.nan)

        logging.info('Data preprocessing completed')
        return final_df
    except KeyError as e:
        logging.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error during preprocessing: %s', e)
        raise


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets."""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(
            raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        logging.debug('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        logging.error('Unexpected error occurred while saving the data: %s', e)
        raise


def main():
    try:
        params = load_params(params_path='params.yaml')
        test_size = params['data_ingestion']['test_size']
        random_state = params['data_ingestion']['random_state']
        data_url = params['data_ingestion']['data_url']
        fileName = params['data_ingestion']['fileName']

        df = load_data(
            data_url=data_url, fileName=fileName)
        cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        final_df = preprocess_data(df, cols)

        train_data, test_data = train_test_split(
            final_df, test_size=test_size, random_state=42)

        save_data(train_data, test_data, data_path='./data')

    except Exception as e:
        logging.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
