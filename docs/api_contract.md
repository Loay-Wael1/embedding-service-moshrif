# API Contract Report

> Generated from actual code inspection. This document reflects the final production API surface for Hugging Face Spaces deployment.

---

## Endpoint Summary Table

| Method | Path | Purpose | Flutter/Web? | Publicly Documented? | Group |
|--------|------|---------|--------------|----------------------|-------|
| GET | `/health` | Liveness check | No | **Yes** | C - Metadata |
| GET | `/legal-info` | Legal RAG service metadata | **Yes (Init)** | **Yes** | C - Metadata |
| POST | `/chat` | Main unified Q&A endpoint | **Yes (Main)** | **Yes** | A - Public |
| GET | `/info` | Embedding model metadata | No | No (Hidden) | B - Debug |
| POST | `/embed` | Embed single text | No | No (Hidden) | B - Debug |
| POST | `/embed/batch` | Embed multiple texts | No | No (Hidden) | B - Debug |
| POST | `/legal-answer` | Internal alias / Advanced debug | No | No (Hidden) | B - Debug |

> **Note:** The old `/ask-legal` endpoint has been completely removed to clean up the API surface.

---

## Detailed Endpoint Documentation

### A) Primary Public Endpoint

#### POST `/chat`

**Purpose:** The primary, unified endpoint for the Legal RAG application. It automatically handles intent classification, legal domain detection, retrieval, source sufficiency checking, and Gemini generation.

**Group:** A - Public App API

**Used by Flutter:** **Yes — This is the ONLY endpoint Flutter/Web should use for queries.**

**Request schema:** `ChatRequest`

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `query` | string | **Yes** | — | The user's message or question |
| `session_id` | string \| null | No | `null` | Optional for future multi-turn tracking |
| `conversation_id` | string \| null | No | `null` | Optional for future multi-turn tracking |

**Minimum Flutter request:**
```json
{
  "query": "ما هي أحكام عقد العمل الفردي؟"
}
```

**Response schema:** `LegalAnswerResponse`

| Field | Type | Always present | Notes |
|-------|------|---------------|-------|
| `query` | string | Yes | Echo of input |
| `answer_mode` | AnswerMode | Yes | `identity`, `conversation`, `non_legal`, `insufficient`, `external_assisted`, `grounded`, `assisted` |
| `final_answer` | string | Yes | **Main answer to display in UI** |
| `warning` | string \| null | Yes | User-facing warning (if any) |
| `internal_sources` | list[SourceCitation] | Yes | Law article citations (if any) |
| `external_sources` | list[SourceCitation] | Yes | External citations (rare) |

**Notes for Flutter/Web Developers:**
- You do **not** need to send `legal_domain`, `law_number`, `top_k`, or any retrieval parameters. The backend intent router detects the domain and adjusts retrieval automatically.
- If the user sends greetings, the backend responds locally without calling Qdrant or Gemini (`answer_mode="conversation"`).
- If the user asks about the bot identity, it responds locally (`answer_mode="identity"`).
- If the user asks an ambiguous question, it will ask for clarification without querying the database (`answer_mode="insufficient"`).
- If the query is related to family/personal status law outside the internal Egyptian law corpus, it will be answered via LLM directly (`answer_mode="external_assisted"`).

---

### B) Metadata Endpoints

#### GET `/legal-info`

**Purpose:** Returns Legal RAG service metadata, including supported answer modes and internal domains. This is lightweight and does not load Qdrant or Gemini.

**Group:** C - Metadata

**Used by Flutter:** **Yes — For application initialization.**

**Response snippet:**
```json
{
  "service": "almostashar-legal-rag",
  "app_name": "المستشار",
  "status": "ok",
  "answer_modes": ["identity", "conversation", "grounded", "assisted", "external_assisted", "insufficient", "non_legal"],
  "supported_internal_domains": ["labor_law", "civil_law", "criminal_law", "constitutional_law"]
}
```

#### GET `/health`

**Purpose:** Standard health/liveness probe for load balancers and Hugging Face Spaces.
**Used by Flutter:** No.

---

### C) Hidden Debug Endpoints (Do Not Use in UI)

These endpoints remain active on the backend for internal debugging and indexing workflows but are explicitly hidden from the OpenAPI (`/docs`) schema to prevent confusion.

- **`POST /legal-answer`**: Advanced RAG querying allowing manual overrides (e.g., forcing a `legal_domain` or `top_k`).
- **`POST /embed`**: Generate a single vector embedding directly from the BGE-M3 model.
- **`POST /embed/batch`**: Generate batch vector embeddings.
- **`GET /info`**: View the internal configuration of the FlagEmbedding backend.

---

## Final Recommendation & Integration Rules

1. **Use `/chat` Exclusively**: Update all Flutter and Web client code to point to `POST /chat`. Send only `{"query": "user text"}`.
2. **Ignore Retrieval Parameters**: The backend fully controls its own RAG loop. The frontend is strictly a presentation layer.
3. **Graceful UI Badges**: Use `answer_mode` to drive UI elements (e.g., show a "Grounded" badge, an "AI Generated" badge for `external_assisted`, or a standard chat bubble for `conversation`).
4. **Ignore Debug Data**: Fields like `llm`, `retrieval_summary`, and `retrieval_result` are stripped in production by default. Do not write frontend logic that depends on them.
