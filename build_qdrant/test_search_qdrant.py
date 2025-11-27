import json
from pathlib import Path

import requests
from qdrant_client import QdrantClient

EMBEDDING_URL = "http://127.0.0.1:8000/embed"
COLLECTION_NAME = "moshrif_knowledge"
VECTOR_SIZE = 1024

DATA_PATH = Path("Moshrif_Knowledge.json")

with DATA_PATH.open("r", encoding="utf-8") as f:
    RAW_DATA = json.load(f)

VIDEOS_BY_ID = {item["id"]: item for item in RAW_DATA}


def get_embedding(text: str):
    payload = {"text": text}
    resp = requests.post(EMBEDDING_URL, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    emb = data["embedding"]

    if len(emb) != VECTOR_SIZE:
        raise ValueError(
            f"Expected embedding of size {VECTOR_SIZE}, got {len(emb)}"
        )
    return emb


def extract_context_around_chunk(video_id, chunk_text, before=1000, after=1500):
    """
    هنا بنجيب الكونتكست من النص الكامل للفيديو:
    - نلاقي مكان الـ chunk في الـ transcript
    - نرجع حتة كبيرة قبله وبعده
    """

    video = VIDEOS_BY_ID.get(video_id)
    if not video:
        return chunk_text 

    full_content = video.get("content", "")
    if not full_content:
        return chunk_text

    probe = chunk_text[:120].strip()
    idx = full_content.find(probe)
    if idx == -1:
        probe = chunk_text[:60].strip()
        idx = full_content.find(probe)

    if idx == -1:
    
        return chunk_text

    start = max(0, idx - before)
    end = min(len(full_content), idx + len(chunk_text) + after)

    return full_content[start:end]


def search(query: str,
           top_points: int = 30,   
           ):
    client = QdrantClient(path="qdrant_db")

    query_vector = get_embedding(query)

    res = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_points,
        with_payload=True,
    )

    hits = getattr(res, "points", res)

    print("=" * 80)
    print(f"Query: {query.strip()}")
    print("-" * 80)

    if not hits:
        print(" NO DATA FROM Qdrant")
        return

    best_hit = max(hits, key=lambda h: getattr(h, "score", 0.0))
    payload = getattr(best_hit, "payload", {}) or {}

    video_id = payload.get("video_id")
    filename = payload.get("filename")
    telegram_url = payload.get("telegram_url")
    chunk_index = payload.get("chunk_index")
    chunk_text = payload.get("content", "")

    score = getattr(best_hit, "score", 0.0)

    big_context = extract_context_around_chunk(
        video_id,
        chunk_text,
        before=1200,   
        after=2000,
    )

    print(f"BEST HIT:")
    print(f"  score      : {score:.4f}")
    print(f"  video_id   : {video_id}")
    print(f"  filename   : {filename}")
    print(f"  telegram   : {telegram_url}")
    print(f"  chunk_idx  : {chunk_index}")
    print("-" * 80)

    print("CONTEXT AROUND MATCH :")
    print(big_context)
    print("-" * 80)

    print("TOP 3 RAW CHUNKS :")
    for i, h in enumerate(hits[:3], start=1):
        p = getattr(h, "payload", {}) or {}
        txt = p.get("content", "").replace("\n", " ")
        print(f"[#{i}] score={h.score:.4f}, video_id={p.get('video_id')}, chunk_idx={p.get('chunk_index')}")
        print(f"     {txt[:250]}...")
        print("-" * 60)


if __name__ == "__main__":
    queries = [
        "ازاي مصر تبقى الهند في تكنولوجيا المعلومات؟",
        "ازاي نختار شغلانة في البرمجة؟",
    ]

    for q in queries:
        search(q)
