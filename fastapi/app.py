import time
import logging
from pathlib import Path
from fastapi import Request
from fastapi.responses import Response
from schema.user_input import User_Input
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
from prometheus_client import generate_latest
from fastapi.templating import Jinja2Templates
from schema.prediction_response import Prediction_Response
from predict import load_model, predict_output, get_model_version
from metrics import MODEL_INFO, MODEL_LOADED, PREDICTION_REQUESTS, PREDICTION_SUCCESS, PREDICTION_FAILURE, PREDICTION_RESULT, PREDICTION_LATENCY


# ------------------------------------------------------------------------------
# Startup / Shutdown
# ------------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):

    try:

        load_model()

        MODEL_LOADED.set(1)

        MODEL_INFO.info({
            "version": str(get_model_version())
        })

    except Exception as e:

        MODEL_LOADED.set(0)

        logging.exception(
            "Model loading failed"
        )

        raise

    yield

    print("Application shutdown")


# ------------------------------------------------------------------------------
# FastAPI App
# ------------------------------------------------------------------------------

app = FastAPI(
    title="Diabetes Prediction API",
    version="1.0",
    lifespan=lifespan
)
BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

templates = Jinja2Templates(directory=BASE_DIR / "templates")
# ------------------------------------------------------------------------------
# Home
# ------------------------------------------------------------------------------


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )

# ------------------------------------------------------------------------------
# Health Check
# ------------------------------------------------------------------------------


@app.get("/health")
def health():

    try:
        load_model()

        return {
            "status": "OK",
            "version": get_model_version(),
            "model_loaded": True
        }
    except Exception as e:
        return {
            "status": "FAILED",
            "model_loaded": False,
            "error": str(e)
        }

# ------------------------------------------------------------------------------
# Prediction Endpoint
# ------------------------------------------------------------------------------


@app.post(
    "/predict",
    response_model=Prediction_Response
)
def predict_diabetes(data: User_Input):

    PREDICTION_REQUESTS.inc()

    start_time = time.perf_counter()

    try:

        user_input = {
            "Pregnancies": data.pregnancies,
            "Glucose": data.glucose,
            "BloodPressure": data.blood_pressure,
            "SkinThickness": data.skin_thickness,
            "Insulin": data.insulin,
            "BMI": data.bmi,
            "DiabetesPedigreeFunction": data.diabetes_pedigree_function,
            "Age": data.age
        }

        response = predict_output(user_input)

        PREDICTION_SUCCESS.inc()

        prediction = response.get("prediction")

        if prediction == 1:

            PREDICTION_RESULT.labels(
                result="diabetes"
            ).inc()

        else:

            PREDICTION_RESULT.labels(
                result="no_diabetes"
            ).inc()

        return response

    except Exception as e:

        PREDICTION_FAILURE.inc()

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:

        PREDICTION_LATENCY.observe(
            time.perf_counter() - start_time
        )

# ------------------------------------------------------------------------------
# Prometheus Metrics Endpoint
# ------------------------------------------------------------------------------


@app.get(
    "/metrics",
    include_in_schema=False
)
def metrics():

    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )
