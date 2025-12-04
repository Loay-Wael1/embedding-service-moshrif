"""
Hierarchical Retrieval Search
==============================
Implements 3-layer retrieval with thresholds and priority logic:
  1. Title layer (threshold: 0.65)
  2. Filename layer (threshold: 0.60)
  3. Content layer (threshold: 0.55)
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests
from qdrant_client import QdrantClient, models

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

QDRANT_PATH = "qdrant_db_hierarchical"
EMBEDDING_URL = "http://127.0.0.1:8000/embed"
VECTOR_SIZE = 1024
COLLECTION_NAME = "moshrif_knowledge_v3"

# Thresholds
TITLE_THRESHOLD = 0.65
FILENAME_THRESHOLD = 0.60
CONTENT_THRESHOLD = 0.55




# ──────────────────────────────────────────────────────────────────────────────
# Embedding Helper
# ──────────────────────────────────────────────────────────────────────────────

def get_embedding(text: str) -> List[float]:
    """Get 1024-dim embedding from local API."""
    payload = {"text": text}
    try:
        resp = requests.post(EMBEDDING_URL, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["embedding"]
    except Exception as e:
        print(f"❌ Error getting embedding: {e}")
        raise


# ──────────────────────────────────────────────────────────────────────────────
# Layer-Specific Searches
# ──────────────────────────────────────────────────────────────────────────────

def search_layer(
    client: QdrantClient,
    query_vector: List[float],
    embedding_type: str,
    limit: int = 10
) -> List[models.ScoredPoint]:
    """Search in a specific embedding layer."""
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="embedding_type",
                    match=models.MatchValue(value=embedding_type)
                )
            ]
        ),
        limit=limit,
        with_payload=True,
        with_vectors=False
    )
    
    # Handle both old and new API
    if hasattr(results, 'points'):
        return results.points
    return results


def get_video_chunks(
    client: QdrantClient,
    video_id: int,
    query_vector: List[float],
    limit: int = 10
) -> List[models.ScoredPoint]:
    """Get all content chunks from a specific video, ranked by similarity."""
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="video_id",
                    match=models.MatchValue(value=video_id)
                ),
                models.FieldCondition(
                    key="embedding_type",
                    match=models.MatchValue(value="content")
                )
            ]
        ),
        limit=limit,
        with_payload=True,
        with_vectors=False
    )
    
    if hasattr(results, 'points'):
        return results.points
    return results


def get_video_chunks_natural_order(
    client: QdrantClient,
    video_id: int,
    limit: int = 1000
) -> List[models.Record]:
    """Get all content chunks from a specific video, ordered by chunk_id (natural order)."""
    results = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="video_id",
                    match=models.MatchValue(value=video_id)
                ),
                models.FieldCondition(
                    key="embedding_type",
                    match=models.MatchValue(value="content")
                )
            ]
        ),
        limit=limit,
        with_payload=True,
        with_vectors=False
    )
    
    # results is a tuple: (points, next_page_offset)
    points = results[0] if isinstance(results, tuple) else results
    
    # Sort by chunk_id to get natural document order
    sorted_points = sorted(points, key=lambda p: p.payload.get("chunk_id", 0))
    
    return sorted_points


# ──────────────────────────────────────────────────────────────────────────────
# Result Formatting
# ──────────────────────────────────────────────────────────────────────────────

def format_chunk_result(point: models.ScoredPoint) -> Dict[str, Any]:
    """Convert Qdrant point to result dict."""
    payload = point.payload or {}
    return {
        "video_id": payload.get("video_id"),
        "filename": payload.get("filename"),
        "telegram_url": payload.get("telegram_url"),
        "chunk_id": payload.get("chunk_id"),
        "chunk_title": payload.get("chunk_title"),
        "chunk_content": payload.get("chunk_content"),
        "similarity": round(point.score, 4)
    }


def format_chunk_result_from_record(point: models.Record) -> Dict[str, Any]:
    """Convert Qdrant Record to result dict (no similarity score - natural order)."""
    payload = point.payload or {}
    return {
        "video_id": payload.get("video_id"),
        "filename": payload.get("filename"),
        "telegram_url": payload.get("telegram_url"),
        "chunk_id": payload.get("chunk_id"),
        "chunk_title": payload.get("chunk_title"),
        "chunk_content": payload.get("chunk_content"),
        "similarity": None  # No score in natural order mode
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main Search Function
# ──────────────────────────────────────────────────────────────────────────────

def search_query(
    query: str,
    top_k: int = 5,
    client: Optional[QdrantClient] = None
) -> Dict[str, Any]:
    """
    Hierarchical search with 3-layer retrieval.
    
    Args:
        query: User query in Arabic
        top_k: Maximum number of results to return
        client: Qdrant client (will create new if None)
    
    Returns:
        {
            "query": str,
            "retrieval_mode": str,
            "scores": {"title": float, "filename": float, "content": float},
            "results": [...]
        }
    """
    if client is None:
        client = QdrantClient(path=QDRANT_PATH)
    
    # ──────────────────────────────────────────────────────────────────────────
    # Step 1: Get query embedding
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"🔍 Query: {query}")
    print(f"{'='*80}")
    
    query_vector = get_embedding(query)
    
    # ──────────────────────────────────────────────────────────────────────────
    # Step 2: Search all 3 layers
    # ──────────────────────────────────────────────────────────────────────────
    print("\n📊 Searching across 3 layers...")
    
    title_results = search_layer(client, query_vector, "title", limit=10)
    filename_results = search_layer(client, query_vector, "filename", limit=10)
    content_results = search_layer(client, query_vector, "content", limit=10)
    
    # Get best matches
    best_title = title_results[0] if title_results else None
    best_filename = filename_results[0] if filename_results else None
    best_content = content_results[0] if content_results else None
    
    title_score = best_title.score if best_title else 0.0
    filename_score = best_filename.score if best_filename else 0.0
    content_score = best_content.score if best_content else 0.0
    
    print(f"\n📈 Scores:")
    print(f"   Title:    {title_score:.4f} {'✅' if title_score >= TITLE_THRESHOLD else '❌'}")
    print(f"   Filename: {filename_score:.4f} {'✅' if filename_score >= FILENAME_THRESHOLD else '❌'}")
    print(f"   Content:  {content_score:.4f} {'✅' if content_score >= CONTENT_THRESHOLD else '❌'}")
    
    # ──────────────────────────────────────────────────────────────────────────
    # Step 3: Apply threshold logic and priority
    # ──────────────────────────────────────────────────────────────────────────
    results = []
    retrieval_mode = "no_strong_match"
    
    # ──────────────────────────────────────────────────────────────────────────
    # CASE 1: Title Match - جيب أفضل 5 chunks من كل الفيديوهات
    # ──────────────────────────────────────────────────────────────────────────
    if (title_score >= TITLE_THRESHOLD and 
        title_score >= filename_score and 
        title_score >= content_score):
        
        print(f"\n✨ Mode: BY TITLE (score={title_score:.4f})")
        retrieval_mode = "by_title"
        
        # ابحث في كل الفيديوهات (content layer) وجيب أفضل 5 chunks
        all_content_results = search_layer(client, query_vector, "content", limit=top_k)
        
        for point in all_content_results:
            results.append(format_chunk_result(point))
        
        print(f"   📊 Returned: Top {len(results)} chunks from all videos (based on content similarity)")
    
    # ──────────────────────────────────────────────────────────────────────────
    # CASE 2: Filename Match - رجّع كل الـ chunks للفيديو
    # ──────────────────────────────────────────────────────────────────────────
    elif (filename_score >= FILENAME_THRESHOLD and 
          filename_score >= content_score):
        
        print(f"\n✨ Mode: BY FILENAME (score={filename_score:.4f})")
        retrieval_mode = "by_filename"
        
        # Get ALL chunks from this video in natural order (by chunk_id)
        video_id = best_filename.payload["video_id"]
        
        # جيب كل الـ chunks مرتبين بترتيبهم الطبيعي (حسب chunk_id)
        all_chunks = get_video_chunks_natural_order(client, video_id, limit=1000)
        
        # رجّع كل الـ chunks بترتيبها الطبيعي (no top_k restriction here)
        for point in all_chunks:
            results.append(format_chunk_result_from_record(point))
        
        print(f"   📊 Returned: ALL {len(results)} chunks from video (in natural order)")
    
    # ──────────────────────────────────────────────────────────────────────────
    # CASE 3: Content Match - رجّع الـ chunk فقط (بدون سياق)
    # ──────────────────────────────────────────────────────────────────────────
    elif content_score >= CONTENT_THRESHOLD:
        
        print(f"\n✨ Mode: BY CONTENT (score={content_score:.4f})")
        retrieval_mode = "by_content"
        
        # Get ONLY the matched chunk (no context)
        main_chunk = format_chunk_result(best_content)
        results.append(main_chunk)
        
        print(f"   📊 Returned: 1 chunk only")
    
    # Fallback: No strong match
    else:
        print(f"\n⚠️  Mode: NO STRONG MATCH (using content fallback)")
        retrieval_mode = "no_strong_match"
        
        # Return top-k from content layer anyway
        for point in content_results[:top_k]:
            results.append(format_chunk_result(point))
    
    # ──────────────────────────────────────────────────────────────────────────
    # Step 4: Format and return
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n✅ Returning {len(results)} results")
    
    return {
        "query": query,
        "retrieval_mode": retrieval_mode,
        "scores": {
            "title": round(title_score, 4),
            "filename": round(filename_score, 4),
            "content": round(content_score, 4)
        },
        "results": results
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main (for testing)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    client = QdrantClient(path=QDRANT_PATH)
    
    # Test queries
    test_queries = [
       " هل الذكاء الاصطناعي هيستبدل المبرمجين ؟"
    ]
    
    for query in test_queries:
        result = search_query(query, top_k=3, client=client)
        
        print(f"\n\n{'='*80}")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"{'='*80}\n")
