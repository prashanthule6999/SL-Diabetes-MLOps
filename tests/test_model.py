# load test + signature test + performance test
import os
import pickle
import mlflow
import unittest
import pandas as pd
from src.helper_func.utility import load_params, setup_mlflow
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set up DagsHub credentials for MLflow tracking
        params = load_params("params.yaml")
        setup_mlflow(params)

        # Load the new model from MLflow model registry
        cls.new_model_name = params["mlflow"]["model_name"]

        cls.new_model_uri = f"models:/{cls.new_model_name}@staging"
        cls.new_model = mlflow.sklearn.load_model(cls.new_model_uri)

        # Load the artifact
        with open('models/model_artifact.pkl', 'rb') as f:
            cls.artifact = pickle.load(f)
        cls.best_threshold = cls.artifact["threshold"]

        # Load holdout test data
        cls.holdout_data = pd.read_csv('data/interim/test_processed.csv')

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_model)

    def test_model_signature(self):
        input_df = pd.DataFrame([{
            "Pregnancies": 6,
            "Glucose": 148,
            "BloodPressure": 72,
            "SkinThickness": 35,
            "Insulin": 0,
            "BMI": 33.6,
            "DiabetesPedigreeFunction": 0.627,
            "Age": 50
        }])

        prediction = self.new_model.predict(input_df)

        # Verify input schema
        self.assertEqual(input_df.shape, (1, 8))

        expected_columns = [
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age",
        ]
        self.assertListEqual(list(input_df.columns), expected_columns)

        # Verify output
        self.assertEqual(prediction.shape, (1,))
        self.assertIn(prediction[0], [0, 1])

    def test_model_performance(self):
        # Extract features and labels from holdout test data
        X_holdout = self.holdout_data.iloc[:, :-1]
        y_holdout = self.holdout_data.iloc[:, -1]

        # Predict using the MLflow model (classifier only)
        y_prob = self.new_model.predict_proba(X_holdout)[:, 1]
        y_pred = (y_prob >= self.best_threshold).astype(int)

        # Calculate performance metrics for the new model
        accuracy_new = accuracy_score(y_holdout, y_pred)
        precision_new = precision_score(y_holdout, y_pred, zero_division=0)
        recall_new = recall_score(y_holdout, y_pred, zero_division=0)
        f1_new = f1_score(y_holdout, y_pred, zero_division=0)

        # Define expected thresholds for the performance metrics
        expected_accuracy = 0.70
        expected_precision = 0.55
        expected_recall = 0.80
        expected_f1 = 0.65

        print(
            f"\nThreshold : {self.best_threshold:.4f}"
            f"\nAccuracy : {accuracy_new:.4f}"
            f"\nPrecision: {precision_new:.4f}"
            f"\nRecall   : {recall_new:.4f}"
            f"\nF1 Score : {f1_new:.4f}"
        )

        # Assert that the new model meets the performance thresholds
        self.assertGreaterEqual(accuracy_new, expected_accuracy,
                                f'Accuracy should be at least {expected_accuracy}')
        self.assertGreaterEqual(precision_new, expected_precision,
                                f'Precision should be at least {expected_precision}')
        self.assertGreaterEqual(
            recall_new, expected_recall, f'Recall should be at least {expected_recall}')
        self.assertGreaterEqual(f1_new, expected_f1,
                                f'F1 score should be at least {expected_f1}')


if __name__ == "__main__":
    unittest.main()
