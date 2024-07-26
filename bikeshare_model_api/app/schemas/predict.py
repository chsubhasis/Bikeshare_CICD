from typing import Any, List, Optional
import datetime

from pydantic import BaseModel
from bikeshare_model.processing.validation import DataInputSchema


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    # predictions: Optional[List[int]]
    predictions: Optional[int]


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "dteday": "2012-11-05",  # datetime.datetime.strptime("2012-11-05", "%Y-%m-%d"),
                        "season": "winter",
                        "hr": "6am",
                        "holiday": "No",
                        "weekday": "Mon",
                        "workingday": "Yes",
                        "weathersit": "Mist",
                        "temp": 6.10,
                        "atemp": 3.0014,
                        "hum": 19.0012,
                        "windspeed": 19.0012,
                        "yr": 2012,
                        "mnth": "November",
                    }
                ]
            }
        }
