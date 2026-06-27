from pydantic import BaseModel, Field
from typing import Annotated, Dict, Optional


class Prediction_Response(BaseModel):

    prediction: Annotated[
        int,
        Field(
            ...,
            description="Predicted class (0 = No Diabetes, 1 = Diabetes)",
            examples=[1]
        )
    ]

    confidence: Annotated[
        Optional[float],
        Field(
            default=None,
            description="Model confidence score",
            examples=[0.8421]
        )
    ]

    class_probabilities: Annotated[
        Optional[Dict[str, float]],
        Field(
            default=None,
            description="Probability distribution across all classes",
            examples=[{
                "No Diabetes": 0.1579,
                "Diabetes": 0.8421
            }]
        )
    ]