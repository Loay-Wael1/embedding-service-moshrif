import json
import time
from pathlib import Path

import requests
from qdrant_client import QdrantClient, models


DATA_PATH = Path("Moshrif_Knowledge.json")

COLLECTION_NAME = "moshrif_knowledge"

VECTOR_SIZE = 1024  

EMBEDDING_URL = "http://127.0.0.1:8000/embed"

BATCH_SIZE = 32



def iter_chunks(text: str, max_chars: int = 800, overlap: int = 100):
    
    text = text.strip()
    if not text:
        return

    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")

    text_len = len(text)
    start = 0

    while start < text_len:
        end = min(start + max_chars, text_len)
        chunk = text[start:end].strip()
        if chunk:
            yield chunk

        if end >= text_len:
            break

        start = end - overlap


def get_embedding(text: str):
    
    payload = {"text": text}
    resp = requests.post(EMBEDDING_URL, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    emb = data["embedding"]

    if len(emb) != VECTOR_SIZE:
        raise ValueError(f"Expected embedding of size {VECTOR_SIZE}, got {len(emb)}")

    return emb



def init_qdrant():
  
    client = QdrantClient(path="qdrant_db") 
   
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=VECTOR_SIZE,
            distance=models.Distance.COSINE,  
        ),
    )

    return client



def build_index():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    print(f"Loading data from {DATA_PATH} ...")
    with DATA_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Total records (videos): {len(data)}")

    client = init_qdrant()
    print("Qdrant collection ready ")

    points_batch = []
    global_point_id = 1

    for video in data:
        vid_id = video.get("id")
        filename = video.get("filename")
        telegram_url = video.get("telegram_url")
        content = video.get("content", "")

        if not content.strip():
            continue

        print(f"Processing video {vid_id} ({filename}) ...")

        for idx, chunk in enumerate(iter_chunks(content, max_chars=800, overlap=100)):
            emb = get_embedding(chunk)

            payload = {
                "video_id": vid_id,
                "filename": filename,
                "telegram_url": telegram_url,
                "chunk_index": idx,
                "content": chunk,
            }

            point = models.PointStruct(
                id=global_point_id,
                vector=emb,
                payload=payload,
            )

            points_batch.append(point)
            global_point_id += 1

            if len(points_batch) >= BATCH_SIZE:
                client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points_batch,
                )
                print(f"Inserted {len(points_batch)} points (latest point_id={global_point_id - 1})")
                points_batch = []

    if points_batch:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points_batch,
        )
        print(f"Inserted final batch: {len(points_batch)} points")

    print(" Finished building Qdrant index!")


if __name__ == "__main__":
    start = time.time()
    build_index()
    print(f"Done in {time.time() - start:.1f} seconds")