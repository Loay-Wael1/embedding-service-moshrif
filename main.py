import re
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from model_loader import embedding_model


# ──────────────────────────────────────────────────────────────────────────────
# Arabic Text Normalization Functions
# ──────────────────────────────────────────────────────────────────────────────

def remove_diacritics(text: str) -> str:
    """Remove Arabic diacritics (تشكيل)."""
    arabic_diacritics = re.compile(r'[\u064B-\u0652\u0670]')
    return arabic_diacritics.sub('', text)


def normalize_alef(text: str) -> str:
    """Normalize all Alef variants to simple Alef (ا)."""
    return re.sub('[إأآ]', 'ا', text)


def normalize_yaa(text: str) -> str:
    """Normalize Yaa variants (ى → ي)."""
    return text.replace('ى', 'ي')


def remove_special_chars(text: str) -> str:
    """Remove underscores, hyphens, and normalize whitespace."""
    text = text.replace('_', ' ')
    text = text.replace('-', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def normalize_arabic_text(text: str) -> str:
    """
    Apply all normalization steps to Arabic text.
    This ensures queries match the normalized data in the database.
    """
    if not text or not isinstance(text, str):
        return text
    
    # 1. Remove diacritics
    text = remove_diacritics(text)
    
    # 2. Normalize Alef
    text = normalize_alef(text)
    
    # 3. Normalize Yaa
    text = normalize_yaa(text)
    
    # 4. Remove special characters
    text = remove_special_chars(text)
    
    return text

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
    
    # Normalize Arabic text before embedding
    normalized_text = normalize_arabic_text(request.text)
    
    # Generate embedding for normalized text
    embedding = embedding_model.embed(normalized_text)
    
    return EmbedResponse(embedding=embedding)
