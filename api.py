from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from nlp_pipeline import get_pipeline


# ---------------------------------------------------------
# FASTAPI APPLICATION
# ---------------------------------------------------------

app = FastAPI(
    title="NLP Pipeline REST API",
    description=(
        "REST API for text preprocessing, "
        "entity extraction and sentiment analysis."
    ),
    version="1.0.0"
)


# ---------------------------------------------------------
# REQUEST MODEL
# ---------------------------------------------------------

class TextRequest(BaseModel):

    text: str = Field(
        ...,
        min_length=1,
        description="Input text"
    )


# ---------------------------------------------------------
# HEALTH CHECK
# ---------------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "ok"
    }


# ---------------------------------------------------------
# PREPROCESS ENDPOINT
# ---------------------------------------------------------

@app.post("/preprocess")
def preprocess(
    request: TextRequest
):

    try:

        result = get_pipeline().preprocess(
            request.text
        )

        return result

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )


# ---------------------------------------------------------
# ENTITY ENDPOINT
# ---------------------------------------------------------

@app.post("/entities")
def extract_entities(
    request: TextRequest
):

    try:

        nlp_pipeline = get_pipeline()

        cleaned = nlp_pipeline.clean_text(
            request.text
        )

        result = nlp_pipeline.entities(
            cleaned
        )

        return {
            "text": cleaned,
            "entities": result
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )


# ---------------------------------------------------------
# SENTIMENT ENDPOINT
# ---------------------------------------------------------

@app.post("/sentiment")
def sentiment(
    request: TextRequest
):

    try:

        nlp_pipeline = get_pipeline()

        cleaned = nlp_pipeline.clean_text(
            request.text
        )

        result = nlp_pipeline.sentiment_analysis(
            cleaned
        )

        return {
            "text": cleaned,
            "sentiment": result
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )


# ---------------------------------------------------------
# COMPLETE ANALYSIS ENDPOINT
# ---------------------------------------------------------

@app.post("/analyze")
def analyze(
    request: TextRequest
):

    try:

        result = get_pipeline().analyze(
            request.text
        )

        return result

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )