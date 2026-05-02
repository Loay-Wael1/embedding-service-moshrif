# Al-Mostashar Legal RAG API

Backend API for **"المستشار"** — an Egyptian Legal RAG assistant.

Repository description:
Egyptian Legal RAG API for "المستشار" — BGE-M3 + Qdrant + Gemini backend for grounded legal chatbot answers.

Suggested topics:
`legal-rag` `egyptian-law` `fastapi` `qdrant` `bge-m3` `gemini-api` `flutter-backend` `rag` `legal-ai` `arabic-nlp`

## Short Description

**المستشار** is an Egyptian legal assistant chatbot designed and developed by **Loay Wael**.

This backend answers Egyptian legal questions using Retrieval-Augmented Generation. It retrieves relevant internal legal sources first, checks whether those sources are sufficient, and then uses Gemini through an OpenAI-compatible endpoint to generate answers grounded in the retrieved sources instead of relying only on the LLM.

## Key Features

- Egyptian Legal RAG backend API.
- Hybrid retrieval using dense and sparse-ready retrieval flow.
- Local BGE-M3 embeddings.
- Local Qdrant vector database.
- Source sufficiency gate before answer generation.
- Answer modes: `identity`, `conversation`, `non_legal`, `grounded`, `assisted`, `external_assisted`, `insufficient`.
- Gemini via OpenAI-compatible Chat Completions API.
- Compact mobile-friendly `POST /chat` endpoint.
- Hidden debug/full endpoint: `POST /legal-answer`.
- Safe fallback when the LLM fails.
- Sanitized public errors for Flutter and user-facing clients.
- CORS-ready for Flutter/Web.
- Lazy loading and `/warmup` support.
- In-memory response cache for safe repeated responses.
- Unit tests and API smoke tests.

## Current Legal Corpus

The current internal corpus covers:

- Egyptian Labor Law.
- Egyptian Civil Law.
- Egyptian Penal Code.
- Egyptian Constitution.

Some Egyptian legal topics are currently outside the internal corpus and are handled as `external_assisted`, including personal status and family law topics such as:

- Custody.
- Alimony.
- Divorce.
- Khula.
- Inheritance.
- Visitation.
- Marital movables list.

For `external_assisted`, the API does not claim internal source verification and does not invent article citations.

## Architecture

```text
User Query
-> Intent Router
-> Domain Routing
-> Hybrid Retrieval with Qdrant
-> Source Sufficiency Gate
-> Gemini Answer Layer
-> Compact API Response
```

Important routing behavior:

- `identity`, `conversation`, and `non_legal` responses do not call Qdrant or Gemini.
- `external_assisted` skips internal retrieval and provides a general legal explanation with a clear warning.
- `grounded` answers are generated from internal sources only.
- `assisted` answers separate what came from internal sources from the assisted explanation.
- `insufficient` avoids presenting an unsupported Egyptian legal answer.

## API Endpoints

Public endpoints:

- `GET /health`
- `GET /legal-info`
- `POST /chat`

Flutter should use only:

```http
POST /chat
```

Request:

```json
{
  "query": "ما ضمانات الحرية الشخصية في الدستور المصري؟"
}
```

Response shape:

```json
{
  "answer_mode": "grounded",
  "final_answer": "...",
  "warning": null,
  "sources": [
    {
      "law_name": "دستور جمهورية مصر العربية",
      "article_number": "54",
      "title": "الحرية الشخصية",
      "source_url": "...",
      "legal_domain": "constitutional_law"
    }
  ],
  "llm": {
    "called": true,
    "succeeded": true,
    "provider": "gemini",
    "model": "gemini-2.5-flash"
  }
}
```

Hidden/debug endpoints:

- `POST /legal-answer`
- `POST /embed`
- `POST /embed/batch`
- `GET /info`

## Flutter Integration

```dart
final response = await http.post(
  Uri.parse('$baseUrl/chat'),
  headers: {'Content-Type': 'application/json'},
  body: jsonEncode({'query': userMessage}),
);
```

Flutter integration rules:

- Send `query` only.
- Do not send `legal_domain`, `top_k`, `law_number`, `law_year`, or retrieval parameters.
- The backend handles intent routing, legal domain routing, retrieval, sufficiency checks, and answer generation automatically.

## Environment Variables

```env
GEMINI_API_KEY=your_key_here
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
GEMINI_MODEL=gemini-2.5-flash
API_PORT=8000
PRELOAD_RETRIEVER=false
CHAT_CONCISE_ANSWERS=true
CHAT_RESPONSE_CACHE_SIZE=128
CHAT_ANSWER_TOP_K=3
DEBUG_RESPONSE_METADATA=false
CORS_ALLOW_ORIGINS=*
CORS_ALLOW_CREDENTIALS=false
```

Security notes:

- Put `GEMINI_API_KEY` in environment variables or platform secrets, not in code.
- For Hugging Face Spaces, use **Settings -> Secrets**.
- Do not commit real API keys.

## Local Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run locally:

```powershell
$env:GEMINI_API_KEY="YOUR_KEY"
$env:GEMINI_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
$env:GEMINI_MODEL="gemini-2.5-flash"
$env:API_PORT="8000"
python main.py
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Testing

```powershell
python -m pytest -q
```

Smoke test:

```powershell
python scripts/smoke_test_api.py --base-url http://127.0.0.1:8000
```

## Hugging Face Deployment

This project includes a `Dockerfile`. For Hugging Face Spaces, use a Docker Space and set secrets/configuration in the Space settings.

Recommended Hugging Face variables:

```env
API_PORT=7860
GEMINI_MODEL=gemini-2.5-flash
PRELOAD_RETRIEVER=false
CHAT_CONCISE_ANSWERS=true
```

Set `GEMINI_API_KEY` in Hugging Face **Settings -> Secrets**.

## Performance Notes

- `/health` and `/legal-info` are lightweight.
- BGE-M3 and Qdrant are lazy-loaded.
- `/warmup` can be used to load the retriever before the first user request.
- Repeated safe queries can be served from the in-memory response cache.
- Streaming can be added later as `POST /chat/stream`.

## Security Notes

- No API keys should be committed.
- Public `/chat` does not expose raw provider errors, HTTP status details, quota messages, API key names, or stack traces.
- Debug metadata is hidden unless `DEBUG_RESPONSE_METADATA=true`.
- `qdrant_db_legal/.lock` and `*.lock` files are ignored.

## Project Status

- Stable local tests currently pass.
- Public API is ready for Flutter integration.
- Streaming is planned as a future enhancement.

## Author

Designed and developed by **Loay Wael**.
