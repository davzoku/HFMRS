from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

import model as model

SORT_BY_DOWNLOADS_ASC = "downloads-asc"
SORT_BY_DOWNLOADS_DESC = "downloads-desc"


app = FastAPI()

# Configure CORS
origins = ["http://localhost", "https://huggingface.co"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RecommendModelRequest(BaseModel):
    """
    Request model for recommending models.

    Attributes:
        model_id (str): ID of the model to get recommendations for.
        method (Optional[str], optional): Method for computing similarities. Defaults to "cosine".
        recommend_no (Optional[int], optional): Number of recommended models to return. Defaults to 5.
    """

    model_id: str
    method: Optional[str] = "cosine"
    recommend_no: Optional[int] = (5,)
    sort_by: Optional[str] = "none"


class RecommendModelResponse(BaseModel):
    """
    Response model for recommended models.

    Attributes:
        model_id (str): ID of the recommended model.
        score (float): Score indicating the similarity of the recommended model to the input model.
    """

    model_id: str
    score: float


def load_model():
    """
    Load the machine learning model.

    Returns:
        my_module.model.Model: The machine learning model.
    """
    model_factory = model.Model(
        data_path="../data/processed/data.csv",
        embed_path="../data/embed.npy",
        model_path="../model/gnn.pt",
        mapping_path="../model/model_mapping.json",
        downloads_path="../model/downloads_mapping.json",
    )
    return model_factory


@app.get("/recommend")
async def recommend_models(
    model_id: str,
    method: Optional[str] = "cosine",
    recommend_no: Optional[int] = 5,
    sort_by: Optional[str] = "none",
) -> List[RecommendModelResponse]:
    """
    Recommends models based on the given model_id and method.

    Args:
        model_id (str): The ID of the model to make recommendations for.
        method (Optional[str], optional): The method to use for recommendations. Defaults to "cosine".
        recommend_no (Optional[int], optional): The number of recommendations to return. Defaults to 5.
        sort_by (Optional[str], optional): The sort by method. Defaults to "none".

    Returns:
        List[RecommendModelResponse]: A list of recommended models with their IDs and scores.
    """
    model_fac = load_model()

    # Make recommendations
    if method == "jaccard":
        result = model_fac.get_recommendation(model_id, metric="jaccard").head(
            recommend_no
        )
    elif method == "gnn":
        result = model_fac.get_recommendation(model_id, metric="gnn").head(recommend_no)
        # normalise scores
        result["score"] = result["score"] / np.linalg.norm(result["score"])
    else:
        result = model_fac.get_recommendation(model_id, metric="cosine").head(
            recommend_no
        )

    # Sort results by downloads if sort_by is "downloads"
    switcher = {
        SORT_BY_DOWNLOADS_ASC.lower(): result.sort_values("downloads", ascending=True),
        SORT_BY_DOWNLOADS_DESC.lower(): result.sort_values(
            "downloads", ascending=False
        ),
    }

    result = switcher.get(sort_by.lower(), result)

    formatted_result = [
        RecommendModelResponse(model_id=id, score=score)
        for id, score in zip(result["modelId"], result["score"])
    ]

    return formatted_result
