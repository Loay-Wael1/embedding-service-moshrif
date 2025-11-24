from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from model_loader import embedding_model

app = FastAPI()


class EmbedRequest(BaseModel):
    text: str


class EmbedResponse(BaseModel):
    embedding: List[float]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/embed", response_model=EmbedResponse)
def embed(request: EmbedRequest) -> EmbedResponse:
    if not request.text:
        raise HTTPException(status_code=400, detail="Text must not be empty.")
    embedding = embedding_model.embed(request.text)
    return EmbedResponse(embedding=embedding)
