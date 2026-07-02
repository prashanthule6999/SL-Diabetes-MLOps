# ------------------------------------------------------------------------------
# This code is a model-serving utility that loads the latest MLflow registered model and
# performs predictions in a thread-safe way.
# ------------------------------------------------------------------------------
import os
import mlflow  # loads model from MLflow Model Registry
import logging
import pandas as pd  # converts input data into DataFrame
from threading import Lock # prevents multiple threads from loading model simultaneously
from mlflow import MlflowClient


# Below code block is for production use
# -------------------------------------------------------------------------------------
# Set up DagsHub credentials for MLflow tracking
# -------------------------------------------------------------------------------------

dagshub_token = os.getenv("SL_DIABETES")
if not dagshub_token:
    raise EnvironmentError("SL_DIABETES environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "prashanthule6999"
repo_name = "SL-Diabetes-MLOps"

mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ------------------------------------------------------------------------------
# Model Configuration
# ------------------------------------------------------------------------------
MODEL_NAME = "diabetes_prediction_model"
MODEL_ALIAS = "production"
MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
model = None
model_lock = Lock()

# ------------------------------------------------------------------------------
# Load Model
# ------------------------------------------------------------------------------


def load_model():
    """
    Thread-safe lazy model loading.
    Model is loaded only once during application lifetime.
    """

    global model

    if model is None:
        with model_lock:
            if model is None:

                logging.info(f"Loading MLflow model from: {MODEL_URI}")
                model = mlflow.sklearn.load_model(MODEL_URI)
                logging.info(
                    f"Model loaded successfully. Version={get_model_version()}"
                )

    return model


# ------------------------------------------------------------------------------
# Prediction Function
# ------------------------------------------------------------------------------
def predict_output(user_input: dict):
    """
    Predict diabetes risk from user input.

    Parameters
    ----------
    user_input : dict

    Example:
    {
        "Pregnancies": 6,
        "Glucose": 148,
        "BloodPressure": 72,
        "SkinThickness": 35,
        "Insulin": 0,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 50
    }
    """

    try:
        model_instance = load_model()

        # Convert request payload to DataFrame
        input_df = pd.DataFrame([user_input])

        # data got converted after transformation
        input_df = input_df.astype({
            "Pregnancies": "int64",
            "Glucose": "float64",
            "BloodPressure": "float64",
            "SkinThickness": "float64",
            "Insulin": "float64",
            "BMI": "float64",
            "DiabetesPedigreeFunction": "float64",
            "Age": "int64"
        })

        # Predict class
        prediction = int(model_instance.predict(input_df)[0])

        response = {
            "prediction": prediction,
            "confidence": None,
            "class_probabilities": None
        }

        try:
            probabilities = model_instance.predict_proba(input_df)[0]

            confidence = float(max(probabilities))

            response["confidence"] = round(confidence, 4)

            response["class_probabilities"] = {
                "No Diabetes": round(float(probabilities[0]), 4),
                "Diabetes": round(float(probabilities[1]), 4)
            }

        except Exception as e:
            logging.warning(
                f"predict_proba not available: {e}"
            )

        return response

    except Exception:
        logging.exception("Prediction failed")
        raise

# ------------------------------------------------------------------------------
# Fetch Model version Function
# ------------------------------------------------------------------------------


def get_model_version():

    client = MlflowClient()

    mv = client.get_model_version_by_alias(
        MODEL_NAME,
        MODEL_ALIAS
    )

    return mv.version
