import mlflow
from src.logger import logging
from mlflow import MlflowClient
from src.helper_func.utility import load_params, setup_mlflow


def promote_model():
    """
    Promote the validated staging model to production.

    Workflow:
    1. Get model version assigned to 'staging' alias.
    2. Point 'production' alias to the same version.
    3. Remove the 'staging' alias.
    """

    try:
        # Configure MLflow
        params = load_params("params.yaml")
        setup_mlflow(params)

        client = MlflowClient()
        model_name = params["mlflow"]["model_name"]

        logging.info("Fetching model assigned to 'staging' alias...")

        # Get the model version currently assigned to the staging alias
        staging_model = client.get_model_version_by_alias(
            name=model_name,
            alias="staging"
        )

        version = staging_model.version

        logging.info(
            f"Model '{model_name}' version {version} found with alias 'staging'."
        )

        # Assign production alias
        client.set_registered_model_alias(
            name=model_name,
            alias="production",
            version=version
        )

        logging.info(
            f"'production' alias assigned to version {version}."
        )

        # Remove staging alias
        client.delete_registered_model_alias(
            name=model_name,
            alias="staging"
        )

        logging.info(
            f"'staging' alias removed from version {version}."
        )

        logging.info(
            f"Model '{model_name}' version {version} successfully promoted to Production."
        )

    except Exception as e:
        logging.exception(f"Model promotion failed: {e}")
        raise


if __name__ == "__main__":
    promote_model()
