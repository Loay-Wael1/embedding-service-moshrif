---
title: Al-Mostashar Legal RAG API
emoji: ⚖️
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
---

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
- Automatic Groq fallback via OpenAI-compatible API when Gemini is unavailable.
- Compact mobile-friendly `POST /chat` worker endpoint, protected in production.
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

Public worker endpoints in production:

- `GET /health`
- `GET /legal-info` if `PROTECT_LEGAL_INFO=false`

Protected worker endpoints in production:

```http
POST /chat
```

Flutter must not call the Hugging Face worker directly in production. Flutter should call the C# platform endpoint only:

```http
POST /api/legal-ai/chat
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
  "answer_parts": {
    "intro": "...",
    "section_title": "أهم الضمانات:",
    "bullets": ["...", "..."],
    "legal_basis": "استندت الإجابة إلى المادة 54 من دستور جمهورية مصر العربية.",
    "note": null
  },
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
- Send it to the C# backend, not directly to Hugging Face.
- Use C# `POST /api/legal-ai/chat` in production.
- Do not send `legal_domain`, `top_k`, `law_number`, `law_year`, or retrieval parameters.
- The backend handles intent routing, legal domain routing, retrieval, sufficiency checks, and answer generation automatically.
- Prefer `answer_parts` for mobile rendering when present; fall back to `final_answer` if it is null.

## Environment Variables

```env
GEMINI_API_KEY=your_key_here
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
GEMINI_MODEL=gemini-2.5-flash
GROQ_API_KEY=your_groq_key_here
GROQ_BASE_URL=https://api.groq.com/openai/v1
GROQ_MODEL=llama-3.3-70b-versatile
LLM_PROVIDER_NAME=gemini
LLM_FALLBACK_PROVIDER_NAME=groq
API_PORT=8000
PRELOAD_RETRIEVER=false
CHAT_CONCISE_ANSWERS=true
CHAT_RESPONSE_CACHE_SIZE=128
CHAT_ANSWER_TOP_K=3
DEBUG_RESPONSE_METADATA=false
ENABLE_PUBLIC_DOCS=false
CORS_ALLOW_ORIGINS=*
CORS_ALLOW_CREDENTIALS=false
REQUIRE_INTERNAL_API_TOKEN=false
INTERNAL_API_TOKEN=
INTERNAL_API_TOKEN_HEADER=X-Internal-Service-Token
PROTECT_LEGAL_INFO=false
```

Security notes:

- Put `GEMINI_API_KEY` in environment variables or platform secrets, not in code.
- Put `GROQ_API_KEY` in environment variables or platform secrets to enable automatic fallback.
- Groq is not xAI Grok. The fallback uses Groq's OpenAI-compatible endpoint.
- Flutter does not change: `/chat` still accepts only `{"query":"..."}`.
- In production, Flutter should call the C# backend only. Enable `REQUIRE_INTERNAL_API_TOKEN=true` on the Python service once the C# gateway is configured.
- In production, set `ENABLE_PUBLIC_DOCS=false` so `/docs`, `/redoc`, and `/openapi.json` return 404.
- If Gemini quota is exhausted or unavailable, the backend tries Groq automatically.
- For Hugging Face Spaces, use **Settings -> Secrets**.
- Do not commit real API keys.

### Gemini Primary + Groq Fallback

Secrets:

```env
GEMINI_API_KEY=<gemini key>
GROQ_API_KEY=<groq key>
```

Variables:

```env
LLM_PROVIDER_NAME=gemini
LLM_FALLBACK_PROVIDER_NAME=groq
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
GEMINI_MODEL=gemini-2.5-flash
GROQ_BASE_URL=https://api.groq.com/openai/v1
GROQ_MODEL=llama-3.3-70b-versatile
```

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

The Space repository is intentionally lightweight and contains application code only. Large runtime assets are stored in a separate Hugging Face Dataset repository:

```text
loaywael10/al-mostashar-legal-rag-assets
```

At runtime, the API downloads the required folders on the first legal query or when `/warmup` is called:

- `./model/bge-m3`
- `./qdrant_db_legal`

Do not push `model/bge-m3` or `qdrant_db_legal` directly to the Space repository.

Recommended Hugging Face variables:

```env
API_PORT=7860
LLM_PROVIDER_NAME=gemini
LLM_FALLBACK_PROVIDER_NAME=groq
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
GEMINI_MODEL=gemini-2.5-flash
GROQ_BASE_URL=https://api.groq.com/openai/v1
GROQ_MODEL=llama-3.3-70b-versatile
PRELOAD_RETRIEVER=false
CHAT_CONCISE_ANSWERS=true
HF_ASSETS_REPO_ID=loaywael10/al-mostashar-legal-rag-assets
HF_ASSETS_REPO_TYPE=dataset
HF_ASSETS_REVISION=main
HF_ASSETS_DOWNLOAD_ENABLED=true
REQUIRE_INTERNAL_API_TOKEN=true
INTERNAL_API_TOKEN=<same secret configured in the C# backend>
INTERNAL_API_TOKEN_HEADER=X-Internal-Service-Token
ENABLE_PUBLIC_DOCS=false
PROTECT_LEGAL_INFO=false
```

Set `GEMINI_API_KEY` in Hugging Face **Settings -> Secrets**.
Set `GROQ_API_KEY` in Hugging Face **Settings -> Secrets** to enable fallback.
If the assets dataset is private, also set `HF_TOKEN` as a Space secret with read access.

## Performance Notes

- `/health` and `/legal-info` are lightweight.
- BGE-M3 and Qdrant are lazy-loaded.
- `/warmup` can be used to load the retriever before the first user request.
- Repeated safe queries can be served from the in-memory response cache.
- Streaming can be added later as `POST /chat/stream`.

## Security Notes

- No API keys should be committed.
- Public `/chat` does not expose raw provider errors, HTTP status details, quota messages, API key names, or stack traces.
- In production, `/chat`, `/warmup`, `/legal-answer`, `/embed`, `/embed/batch`, and `/info` require `X-Internal-Service-Token`.
- In production, `/docs`, `/redoc`, and `/openapi.json` are disabled.
- `/health` remains public for platform health checks.
- Debug metadata is hidden unless `DEBUG_RESPONSE_METADATA=true`.
- `qdrant_db_legal/.lock` and `*.lock` files are ignored.

## Project Status

- Stable local tests currently pass.
- Public API is ready for Flutter integration.
- Streaming is planned as a future enhancement.

## Author

Designed and developed by **Loay Wael**.
