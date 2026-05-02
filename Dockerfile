FROM python:3.10-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app/ app/
COPY main.py .
COPY ask_legal_llm.py .
COPY ask_legal.py .
COPY env.example .

# Copy local model and Qdrant index
COPY model/ model/
COPY qdrant_db_legal/ qdrant_db_legal/

# Expose HF Spaces port
EXPOSE 7860

# Single worker for local Qdrant safety
CMD ["python", "main.py"]
