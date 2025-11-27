
````markdown
# Embedding Service (Moshrif Knowledge)

A simple, self-hosted **embedding & semantic search service** for Arabic content, built as part of a graduation project.

This repo provides:

- 🧠 A **FastAPI HTTP service** that generates 1024-dimensional embeddings for Arabic text using a local model.
- 🔍 A **semantic search system** over Mohamed Moshrif transcripts using **Qdrant** (embedded/local vector DB).
- 🧾 A ready JSON dataset of ~253 videos, plus an optional prebuilt Qdrant database.

> ⚠️ This project is pure **Python**. Any backend (.NET, Node, Django, etc.) can consume it via simple HTTP calls.

---

## Overview

This project is a small **RAG-style building block**:

1. **Embedding API**  
   A FastAPI service exposes `POST /embed`, which takes text and returns a 1024-dim embedding vector.  
   Any external backend (web app, .NET API, etc.) calls this endpoint to get embeddings.

2. **Semantic Search over Moshrif Content**  
   Inside `build_qdrant/` you’ll find:
   - `Moshrif_Knowledge.json` → all video transcripts + metadata
   - `build_qdrant_index.py` → builds the Qdrant index:
     - splits transcripts into chunks
     - calls `/embed` to generate embeddings
     - stores vectors in a local Qdrant collection
   - `test_search_qdrant.py` → runs a semantic search:
     - embeds the user query
     - searches Qdrant
     - extracts a **large context** from the original transcript around the best matching chunk

---

## Features

- ✅ Local **FastAPI** service exposing `/embed`
- ✅ **1024-dim embeddings** (tuned for Arabic text)
- ✅ Local **Qdrant embedded mode** (no external DB server needed)
- ✅ **Chunking with overlap** to reduce information loss at boundaries
- ✅ **Rich payloads**: `video_id`, `filename`, `telegram_url`, `chunk_index`, `content`
- ✅ **Context expansion**: return a big chunk of text around the best match
- ✅ Ready dataset (`Moshrif_Knowledge.json`) + **prebuilt Qdrant DB download**

---

## Project Structure

```text
embedding-service-moshrif/
│
├── main.py                      # FastAPI application (/health, /embed)
├── model_loader.py              # Loads embedding model and computes vectors
├── config.py                    # Config (model path, device, etc.)
├── requirements.txt             # Python dependencies
│
├── model/                       # Local embedding model (e.g. BGE-M3)
│   └── bge-m3/                  # Model files (NOT usually committed)
│
├── build_qdrant/
│   ├── Moshrif_Knowledge.json   # ~253 Moshrif video transcripts
│   ├── build_qdrant_index.py    # Build/fill Qdrant collection
│   └── test_search_qdrant.py    # Run semantic search + context expansion
│
├── qdrant_db/                   # Qdrant embedded DB (generated or downloaded)
│
├── .gitignore                   # Ignores venv, qdrant_db, etc.
└── README.md                    # This file
````

---

## Requirements

* **Python** 3.10 (recommended; used in development)
* **pip** (latest recommended)
* OS: Windows / Linux / macOS (tested on Windows)
* Disk space for:

  * Local model (hundreds of MB)
  * Qdrant DB (hundreds of MB depending on data)

Main Python deps (see `requirements.txt`):

* `fastapi`
* `uvicorn`
* `transformers`
* `torch`
* `pydantic`
* `qdrant-client==1.16.1`  ← version used to build and query the index

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Loay-Wael1/embedding-service-moshrif.git
cd embedding-service-moshrif
```

### 2. Create & activate virtual environment

```bash
# Create venv
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install "qdrant-client==1.16.1"
```

---

## Run the Embedding Service

From the project root:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

> ⚠️ Avoid using `--reload` here to prevent loading the model twice in the same process.

Check:

* Health: `http://127.0.0.1:8000/health`
* Swagger docs: `http://127.0.0.1:8000/docs`



---

## Building the Qdrant Index

### Option A: Build locally

> The embedding service must be running on `http://127.0.0.1:8000` before running this script.

From the project root:

```bash
cd build_qdrant
python build_qdrant_index.py
```

What this script does:

1. Reads `Moshrif_Knowledge.json` (all videos).
2. For each video:

   * splits the transcript into chunks (≈800 chars with 100-char overlap)
   * calls `/embed` for each chunk
   * stores the vector + metadata in Qdrant (`../qdrant_db`)
3. When finished, you should see:

```text
Finished building Qdrant index!
Done in XXXX.X seconds
```

### Option B: Use prebuilt database

If you don’t want to wait for all embeddings to be generated:

1. Download the prebuilt Qdrant DB from Google Drive:

   **📥 Prebuilt Qdrant DB**
   [https://drive.google.com/drive/folders/1bEqW2mC-t50Cl8pfYxXJVrZ_j6EtLq8M?usp=sharing](https://drive.google.com/drive/folders/1bEqW2mC-t50Cl8pfYxXJVrZ_j6EtLq8M?usp=sharing)

2. Place the `qdrant_db` folder in the project root:

```text
embedding-service-moshrif/
├── qdrant_db/
├── main.py
├── build_qdrant/
└── ...
```

3. You can now run the search script directly.

---

## Semantic Search (Test Script)

From the project root:

```bash
cd build_qdrant
python test_search_qdrant.py
```

The script:

* Embeds example queries (e.g. `ازاي مصر تبقى الهند في تكنولوجيا المعلومات؟`)
* Queries Qdrant (`moshrif_knowledge` collection) using `QdrantClient(path="qdrant_db")`
* Picks the best match
* Expands context using the original transcript in `Moshrif_Knowledge.json`

Example output:

```text
================================================================================
Query: ازاي نختار شغلانة في البرمجة؟
--------------------------------------------------------------------------------
BEST HIT:
  score      : 0.56
  video_id   : 152
  filename   : ...
  telegram   : https://t.me/...
  chunk_idx  : 12
--------------------------------------------------------------------------------
CONTEXT AROUND MATCH :
[long passage around the relevant part of the video...]
```

---

## How the Qdrant Pipeline Works (Summary)

1. **Data**
   `Moshrif_Knowledge.json` contains records like:

   ```json
   {
     "id": 1,
     "filename": "some_video_name",
     "telegram_url": "https://t.me/...",
     "content": "full Arabic transcript..."
   }
   ```

2. **Chunking**
   `build_qdrant_index.py` uses `iter_chunks(text, max_chars=800, overlap=100)` to split each transcript into overlapping chunks.

3. **Embedding**
   Each chunk is sent to the `/embed` endpoint, and a 1024-dim vector is returned.

4. **Indexing in Qdrant**
   A collection `moshrif_knowledge` is created with:

   * vector size: 1024
   * distance: cosine
     Vectors + metadata are stored using `client.upsert(...)`.

5. **Search**
   `test_search_qdrant.py`:

   * embeds the query
   * calls `client.query_points(...)`
   * selects the best hit
   * extracts a large context window from the original transcript around that chunk.

This is the entire pipeline used for semantic search over Moshrif’s content.

```
::contentReference[oaicite:0]{index=0}
```
