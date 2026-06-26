# data preprocessing

import os
import numpy as np
import pandas as pd
from typing import Tuple
from src.logger import logging
from sklearn.model_selection import train_test_split
from src.helper_func.utility import load_params, load_data_from_csv

def replace_invalid_zeros(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    df[cols] = df[cols].replace(0, np.nan)
    return df


def train_val_test_split(raw_cleaned_data: pd.DataFrame, test_size: float, val_size: float, random_state: int, target_column: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Validation split is used for threshold optimization,
    # preventing the test set from influencing threshold selection.
    train_full, test_full = train_test_split(
        raw_cleaned_data,
        test_size=test_size,
        stratify=raw_cleaned_data[target_column],
        random_state=random_state
    )

    # Split training portion again
    train, val = train_test_split(
        train_full,
        test_size=val_size,   # 0.25 × 0.80 = 0.20 (20% of original dataset)
        stratify=train_full[target_column],
        random_state=random_state
    )

    return train, val, test_full


def main() -> None:
    try:
        # Fetch the data from data/raw
        raw_data = load_data_from_csv('./data/raw/diabetes.csv')

        params = load_params('params.yaml')

        test_size = params["data_preprocessing"]['test_size']
        val_size = params["data_preprocessing"]['val_size']
        random_state = params["data_preprocessing"]['random_state']
        target_column = params["data_preprocessing"]['target_column']
        columns = params["data_preprocessing"]['columns']

        # Transform the data
        raw_cleaned_data = replace_invalid_zeros(raw_data, columns)

        train_data, val_data, test_data = train_val_test_split(raw_cleaned_data, test_size,
                                                               val_size, random_state, target_column)

        # Store the data inside data/interim
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)

        train_data.to_csv(os.path.join(
            data_path, "train_processed.csv"), index=False)
        val_data.to_csv(os.path.join(
            data_path, "val_processed.csv"), index=False)
        test_data.to_csv(os.path.join(
            data_path, "test_processed.csv"), index=False)
        
        logging.info("Train shape: %s", train_data.shape)
        logging.info("Validation shape: %s", val_data.shape)
        logging.info("Test shape: %s", test_data.shape)

        logging.info('Processed data saved to %s', data_path)
    except Exception as e:
        logging.error(
            'Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
