from pydantic import BaseModel, Field, ConfigDict
from typing import Annotated


class User_Input(BaseModel):

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "pregnancies": 6,
                "glucose": 148,
                "blood_pressure": 72,
                "skin_thickness": 35,
                "insulin": 1,
                "bmi": 33.6,
                "diabetes_pedigree_function": 0.627,
                "age": 50
            }
        }
    )

    pregnancies: Annotated[
        int,
        Field(description="Number of pregnancies", ge=0, le=20)
    ]

    glucose: Annotated[
        int,
        Field(description="Plasma glucose concentration", ge=40, le=300)
    ]

    blood_pressure: Annotated[
        int,
        Field(description="Diastolic blood pressure (mm Hg)", ge=20, le=180)
    ]

    skin_thickness: Annotated[
        int,
        Field(description="Triceps skin fold thickness (mm)", ge=0, le=100)
    ]

    insulin: Annotated[
        int,
        Field(description="2-Hour serum insulin (mu U/ml)", ge=0, le=1000)
    ]

    bmi: Annotated[
        float,
        Field(description="Body Mass Index", ge=10, le=80)
    ]

    diabetes_pedigree_function: Annotated[
        float,
        Field(description="Family history score", ge=0, le=5)
    ]

    age: Annotated[
        int,
        Field(description="Age in years", ge=1, le=120)
    ]
