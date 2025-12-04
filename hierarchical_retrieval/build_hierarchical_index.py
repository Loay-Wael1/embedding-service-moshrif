"""
Hierarchical Indexing Script
=============================
Builds a 3-layer Qdrant index:
  1. Filename embeddings (video-level)
  2. Title embeddings (chunk titles)
  3. Content embeddings (chunk contents)
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any

import requests
from qdrant_client import QdrantClient, models

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

DATA_PATH = Path("Moshrif-knowledge-chunks.json")
QDRANT_PATH = "qdrant_db_hierarchical"
EMBEDDING_URL = "http://127.0.0.1:8000/embed"
VECTOR_SIZE = 1024
COLLECTION_NAME = "moshrif_knowledge_v3"
BATCH_SIZE = 64


# ──────────────────────────────────────────────────────────────────────────────
# Embedding Helper
# ──────────────────────────────────────────────────────────────────────────────

def get_embedding(text: str) -> List[float]:
    """Get 1024-dim embedding from local API."""
    if not text or not text.strip():
        # Return zero vector for empty text
        return [0.0] * VECTOR_SIZE
    
    payload = {"text": text}
    try:
        resp = requests.post(EMBEDDING_URL, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        emb = data["embedding"]
        
        if len(emb) != VECTOR_SIZE:
            raise ValueError(f"Expected {VECTOR_SIZE} dims, got {len(emb)}")
        
        return emb
    except Exception as e:
        print(f"❌ Error getting embedding: {e}")
        raise


# ──────────────────────────────────────────────────────────────────────────────
# Index Building
# ──────────────────────────────────────────────────────────────────────────────

def init_collection(client: QdrantClient):
    """Recreate the collection with proper schema."""
    print(f"🔨 Recreating collection: {COLLECTION_NAME}")
    
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=VECTOR_SIZE,
            distance=models.Distance.COSINE,
        ),
    )
    print("✅ Collection created successfully")


def build_hierarchical_index(data: List[Dict[str, Any]], client: QdrantClient):
    """
    Build 3-layer hierarchical index:
      - Layer 1: Filename embeddings (1 per video)
      - Layer 2: Title embeddings (1 per chunk)
      - Layer 3: Content embeddings (1 per chunk)
    """
    points_batch = []
    global_point_id = 1
    
    print(f"\n📊 Processing {len(data)} videos...")
    
    for video_idx, video in enumerate(data, 1):
        video_id = video.get("id")
        filename = video.get("filename", "")
        telegram_url = video.get("telegram_url", "")
        chunks = video.get("chunks", [])
        
        if not chunks:
            print(f"⚠️  Skipping video {video_id}: no chunks")
            continue
        
        print(f"\n[{video_idx}/{len(data)}] Video ID={video_id}, Chunks={len(chunks)}")
        
        # ─────────────────────────────────────────────────────────────────────
        # LAYER 1: Filename Embedding (1 point per video)
        # ─────────────────────────────────────────────────────────────────────
        print(f"  📁 Creating filename embedding...")
        filename_emb = get_embedding(filename)
        
        filename_point = models.PointStruct(
            id=global_point_id,
            vector=filename_emb,
            payload={
                "video_id": video_id,
                "filename": filename,
                "telegram_url": telegram_url,
                "chunk_id": None,
                "chunk_title": "",
                "chunk_content": "",
                "embedding_type": "filename"
            }
        )
        points_batch.append(filename_point)
        global_point_id += 1
        
        # ─────────────────────────────────────────────────────────────────────
        # LAYER 2 & 3: Title + Content Embeddings (2 points per chunk)
        # ─────────────────────────────────────────────────────────────────────
        for chunk in chunks:
            chunk_id = chunk.get("chunk_id")
            title = chunk.get("topicTitle", "").strip()
            content = chunk.get("topicContent", "").strip()
            
            if not content:
                print(f"    ⚠️  Skipping chunk {chunk_id}: empty content")
                continue
            
            # LAYER 2: Title Embedding
            if title:
                title_emb = get_embedding(title)
                
                title_point = models.PointStruct(
                    id=global_point_id,
                    vector=title_emb,
                    payload={
                        "video_id": video_id,
                        "filename": filename,
                        "telegram_url": telegram_url,
                        "chunk_id": chunk_id,
                        "chunk_title": title,
                        "chunk_content": content,
                        "embedding_type": "title"
                    }
                )
                points_batch.append(title_point)
                global_point_id += 1
            
            # LAYER 3: Content Embedding
            content_emb = get_embedding(content)
            
            content_point = models.PointStruct(
                id=global_point_id,
                vector=content_emb,
                payload={
                    "video_id": video_id,
                    "filename": filename,
                    "telegram_url": telegram_url,
                    "chunk_id": chunk_id,
                    "chunk_title": title,
                    "chunk_content": content,
                    "embedding_type": "content"
                }
            )
            points_batch.append(content_point)
            global_point_id += 1
            
            # Batch upsert
            if len(points_batch) >= BATCH_SIZE:
                client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points_batch
                )
                print(f"    ✅ Inserted batch of {len(points_batch)} points")
                points_batch = []
    
    # Final batch
    if points_batch:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points_batch
        )
        print(f"\n✅ Inserted final batch of {len(points_batch)} points")
    
    print(f"\n🎉 Finished indexing! Total points: {global_point_id - 1}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    """Main indexing pipeline."""
    # Load data
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
    
    print(f"📂 Loading data from {DATA_PATH}...")
    with DATA_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"✅ Loaded {len(data)} videos")
    
    # Initialize Qdrant client
    client = QdrantClient(path=QDRANT_PATH)
    
    # Create collection
    init_collection(client)
    
    # Build index
    build_hierarchical_index(data, client)
    
    # Show stats
    collection_info = client.get_collection(COLLECTION_NAME)
    print(f"\n📊 Collection Stats:")
    print(f"   Total points: {collection_info.points_count}")
    print(f"   Vector size: {collection_info.config.params.vectors.size}")
    print(f"   Distance: {collection_info.config.params.vectors.distance}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f"\n⏱️  Total time: {elapsed:.1f} seconds")
