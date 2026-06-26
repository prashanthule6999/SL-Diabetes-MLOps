# data ingestion

import kagglehub
import pandas as pd
from pathlib import Path
from src.logger import logging
from src.helper_func.utility import load_params

def load_data_from_external_src(data_url: str, file_name: str) -> pd.DataFrame:
    """Load data from kaggle site."""
    try:
        path = Path(kagglehub.dataset_download(data_url))
        csv_file = path / file_name
        if not csv_file.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {csv_file}"
            )

        df = pd.read_csv(csv_file)

        if df.empty:
            raise ValueError("Loaded dataset is empty")
        logging.info('Data loaded from %s/%s', data_url, file_name)

        logging.info(
            "Dataset contains. Rows=%d Columns=%d",
            df.shape[0],
            df.shape[1]
        )

        return df

    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error(
            'Unexpected error occurred while loading the data: %s', e)
        raise


def save_data(df: pd.DataFrame, file_name: str, data_path: str) -> None:
    """Save datasets."""
    try:
        raw_data_path = Path(data_path) / "raw"
        raw_data_path.mkdir(parents=True, exist_ok=True)

        output_file = raw_data_path / file_name
        df.to_csv(output_file, index=False)

        logging.info(
            "Dataset saved successfully at %s",
            output_file
        )

    except Exception as e:
        logging.error('Unexpected error occurred while saving the data: %s', e)
        raise


def main() -> None:

    try:
        params = load_params('params.yaml')

        data_url = params["data_ingestion"]['data_url']
        file_name = params["data_ingestion"]['file_name']
        raw_data_path = params["data_ingestion"]["raw_data_path"]

        df = load_data_from_external_src(
            data_url, file_name)

        save_data(df, file_name, raw_data_path)

    except Exception as e:
        logging.error('Failed to complete the data ingestion process: %s', e)
        raise


if __name__ == '__main__':
    main()
