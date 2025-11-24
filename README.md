# 🧠 Embedding Service — BGE-M3 + FastAPI

A lightweight Python API built on **FastAPI** that generates **1024-D embeddings** using the **BAAI/BGE-M3** model.  
Perfect for **RAG systems, vector search, Qdrant, Pinecone, semantic similarity, document retrieval**, etc.

This README guides you from **zero → running the API**, starting even from a clean Windows installation.

---

# 🚀 Features

- 🔥 FastAPI HTTP service
- 🧠 Uses BGE-M3 (state-of-the-art embedding model)
- 🔢 Outputs 1024-dim embedding vectors
- ⚡ CPU mode (default) — reliable on Windows
- 🧱 Clean, modular architecture
- 🔄 Auto-download model from HuggingFace
- 📦 Can also run fully offline (local model folder)
- 🔒 Loads model only **once** in memory

---

# 📌 1. Install Python **3.10**

This project **requires Python 3.10**, because PyTorch + Transformers on Windows  
work best on this version.

Download Python 3.10.12:

🔗 https://www.python.org/downloads/release/python-31012/

During installation:

✔ Add Python to PATH  
✔ Enable all optional features  
✔ Install for all users (optional)

Confirm installation:

python --version
or:
py -3.10 --version

Output should be:
Python 3.10.x

# 📌 2. Enable Windows Developer Mode

This prevents HuggingFace symlink issues and speeds up downloads.

Open RUN:

start ms-settings:developers


Enable:

✔ Developer Mode

# 📌 3. Clone the Repository
git clone https://github.com/Mohamed-ALQarram/Embedding-Service.git
cd Embedding-Service

# 📌 4. Create Virtual Environment (venv)
If you have multiple Python versions (recommended):
py -3.10 -m venv venv

Otherwise:
python -m venv venv


Activate venv:

Windows
venv\Scripts\activate

Linux/Mac
source venv/bin/activate

# 📌 5. Install Requirements
pip install -r requirements.txt

# 📌 6. Configure the Model (Online Download)

Open config.py and set:

MODEL_NAME = "BAAI/bge-m3"
DEVICE = "cpu"


Why CPU?

Stable on all Windows machines

Avoids CUDA errors

Works out-of-the-box

⚠ If you don’t have CUDA → DO NOT use "cuda".

# 📌 7. Run the Service

Start FastAPI:

uvicorn main:app


⚠ Important:
Do NOT use --reload because it will load the model twice.

You should see:

INFO: Uvicorn running on http://127.0.0.1:8000

# 📌 8. Test the API (Swagger UI)

Open:

👉 http://127.0.0.1:8000/docs

Try the /embed endpoint.

Example Input:
{
  "text": "hello world"
}

Example Output:
{
  "embedding": [ ...1024 float values... ]
}

# 📌 9. Offline Mode (Local Model Folder)

If you want to run offline:

Create folder:

model/bge-m3/


Download these files from HuggingFace:

config.json

pytorch_model.bin

tokenizer.json

tokenizer_config.json

sentencepiece.bpe.model

special_tokens_map.json

Update config.py:

MODEL_NAME = "./model/bge-m3"
DEVICE = "cpu"

# 📌 10. Project Structure
Embedding-Service/
│
├── main.py               # FastAPI routes
├── model_loader.py       # Model + tokenizer initialization
├── config.py             # Settings (model + device)
├── requirements.txt
└── model/ (optional)
    └── bge-m3/

# 📌 11. Example cURL Request
curl -X POST "http://127.0.0.1:8000/embed" \
     -H "Content-Type: application/json" \
     -d "{\"text\": \"Hello embedding world\"}"

#📌 12. Troubleshooting
⚠ CUDA not available

Set:

DEVICE = "cpu"

⚠ Model downloads twice

Don’t use --reload.

⚠ Tokenizer errors

Ensure you downloaded all tokenizer files if using offline mode.

⚠ PyTorch errors

Ensure you're using Python 3.10 only.
