import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
# print(sys.path)
from typing import Any

from fastapi import APIRouter, FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app.api import api_router
from app.config import settings

from bikeshare_model.predict import make_prediction

curr_path = str(Path(__file__).parent)

app = FastAPI(
    title=settings.PROJECT_NAME, openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

root_router = APIRouter()

################################# Prometheus related code START ######################################################
import prometheus_client as prom

import pandas as pd
from sklearn.metrics import r2_score

# Metric object of type gauge
r2_metric = prom.Gauge('bikeshare_r2_score', 'R2 score for random 100 test samples')


# LOAD TEST DATA
test_data = pd.read_csv(curr_path + "/test_bikeshare.csv")


# Function for updating metrics
def update_metrics():
    test = test_data.sample(100)
    test_feat = test.drop('cnt', axis=1)
    test_cnt = test['cnt'].values
    test_pred = make_prediction(input_data=test_feat)['predictions']
    #r2 = r2_score(test_cnt, test_pred).round(3)
    r2 = r2_score(test_cnt, test_pred)
    
    r2_metric.set(r2)


@app.get("/metrics")
async def get_metrics():
    update_metrics()
    return Response(media_type="text/plain", content= prom.generate_latest())

################################# Prometheus related code END ######################################################

@root_router.get("/")
def index(request: Request) -> Any:
    """Basic HTML response."""
    body = (
        "<html>"
        "<body style='padding: 10px;'>"
        "<h1>Welcome to the API</h1>"
        "<div>"
        "Check the docs: <a href='/docs'>here</a>"
        "</div>"
        "</body>"
        "</html>"
    )

    return HTMLResponse(content=body)


app.include_router(api_router, prefix=settings.API_V1_STR)
app.include_router(root_router)

# Set all CORS enabled origins
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


if __name__ == "__main__":
    # Use this for debugging purposes only
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
