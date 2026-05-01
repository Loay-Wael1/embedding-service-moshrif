from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import httpx

from app.settings import Settings, settings


class LLMError(RuntimeError):
    pass


class LLMConfigurationError(LLMError):
    pass


class LLMRequestError(LLMError):
    pass


@dataclass(slots=True)
class LLMCompletion:
    content: str
    model: str
    provider: str
    usage: dict[str, Any] | None = None
    raw_response: dict[str, Any] | None = None


class OpenAICompatibleLLMClient:
    """Small OpenAI-compatible Chat Completions client.

    The default configuration targets Gemini's OpenAI-compatible endpoint.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        provider_name: str | None = None,
        timeout_seconds: float | None = None,
        config: Settings | None = None,
    ) -> None:
        active = config or settings
        self.provider_name = provider_name or active.llm_provider_name
        self.api_key = api_key if api_key is not None else active.llm_api_key
        self.base_url = (base_url or active.llm_base_url).rstrip("/")
        self.model = model or active.llm_model
        self.timeout_seconds = timeout_seconds or active.llm_timeout_seconds
        self.max_tokens = active.llm_max_tokens
        self.web_search_enabled = active.llm_web_search_enabled
        self.extra_body = _parse_extra_body(active.llm_extra_body_json)
        self.json_mode = active.llm_json_mode

    def chat_completion(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMCompletion:
        if not self.api_key:
            raise LLMConfigurationError(
                "LLM_API_KEY or GEMINI_API_KEY is not configured. Set it in the environment; do not hard-code it."
            )

        effective_max = max_tokens or self.max_tokens
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": effective_max,
            "stream": False,
        }
        payload.update(self.extra_body)

        # Attempt JSON mode first if configured, fall back if unsupported.
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            data = self._http_post(url, headers, payload)
        except LLMRequestError as exc:
            error_msg = str(exc)
            needs_retry = False
            # If the error indicates response_format is unsupported, retry without it.
            if self.json_mode and _is_response_format_unsupported(error_msg):
                payload.pop("response_format", None)
                needs_retry = True
            # If the error indicates max_completion_tokens is unsupported, fall back to max_tokens.
            if _is_max_completion_tokens_unsupported(error_msg):
                payload.pop("max_completion_tokens", None)
                payload["max_tokens"] = effective_max
                needs_retry = True
            if needs_retry:
                data = self._http_post(url, headers, payload)
            else:
                raise

        content = _extract_message_content(data)
        if not content:
            raise LLMRequestError(f"{self.provider_name} response did not contain assistant content.")

        return LLMCompletion(
            content=content,
            model=str(data.get("model") or self.model),
            provider=self.provider_name,
            usage=data.get("usage") if isinstance(data.get("usage"), dict) else None,
            raw_response=data,
        )

    def _http_post(self, url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            detail = _extract_error_detail(exc.response)
            raise LLMRequestError(
                f"{self.provider_name} returned HTTP {exc.response.status_code}: {detail}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise LLMRequestError(f"{self.provider_name} request timed out.") from exc
        except httpx.RequestError as exc:
            raise LLMRequestError(f"{self.provider_name} request failed: {exc}") from exc
        except ValueError as exc:
            raise LLMRequestError(f"{self.provider_name} returned a non-JSON response.") from exc
        return data


def _is_response_format_unsupported(error_message: str) -> bool:
    """Check whether an LLM error indicates that ``response_format`` is not supported."""
    lower = error_message.lower()
    indicators = (
        "response_format",
        "json_object",
        "json mode",
        "unsupported parameter",
        "not supported",
        "invalid parameter",
        "unknown parameter",
    )
    return any(indicator in lower for indicator in indicators)


def _is_max_completion_tokens_unsupported(error_message: str) -> bool:
    """Check whether an LLM error indicates that ``max_completion_tokens`` is not supported."""
    lower = error_message.lower()
    indicators = (
        "max_completion_tokens",
        "maxcompletiontokens",
    )
    return any(indicator in lower for indicator in indicators)


def _extract_message_content(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0] or {}
    message = first.get("message") or {}
    content = message.get("content", first.get("text", ""))
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts).strip()
    return ""


def _extract_error_detail(response: httpx.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        return response.text[:500]
    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error)
        if error:
            return str(error)
    return str(data)[:500]


def _parse_extra_body(value: str) -> dict[str, Any]:
    if not value.strip():
        return {}
    try:
        data = json.loads(value)
    except json.JSONDecodeError as exc:
        raise LLMConfigurationError("LLM_EXTRA_BODY_JSON must be valid JSON.") from exc
    if not isinstance(data, dict):
        raise LLMConfigurationError("LLM_EXTRA_BODY_JSON must decode to a JSON object.")
    return data


# ---------------------------------------------------------------------------
# JSON parsing and answer-payload validation
# ---------------------------------------------------------------------------


def parse_llm_json(raw: str) -> dict[str, Any]:
    """Parse a raw LLM response string into a Python dict.

    Raises ``ValueError`` with detailed diagnostics when the response
    cannot be parsed as JSON at all.
    """
    text = (raw or "").strip()
    if not text:
        raise ValueError("LLM returned empty response")

    # Strip markdown fences.
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()

    # Attempt 1: direct parse.
    result = _try_json_loads(text)
    if isinstance(result, dict):
        return result

    # Attempt 2: fix literal (unescaped) newlines inside JSON strings.
    fixed = _fix_literal_newlines(text)
    if fixed != text:
        result = _try_json_loads(fixed)
        if isinstance(result, dict):
            return result

    # Attempt 3: balanced-brace extraction.
    balanced = _extract_first_json_object(text)
    if balanced:
        result = _try_json_loads(balanced)
        if isinstance(result, dict):
            return result
        fixed_balanced = _fix_literal_newlines(balanced)
        if fixed_balanced != balanced:
            result = _try_json_loads(fixed_balanced)
            if isinstance(result, dict):
                return result

    # Attempt 4: greedy regex extraction.
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidate = match.group(0)
        result = _try_json_loads(candidate)
        if isinstance(result, dict):
            return result
        fixed_candidate = _fix_literal_newlines(candidate)
        if fixed_candidate != candidate:
            result = _try_json_loads(fixed_candidate)
            if isinstance(result, dict):
                return result

    # All attempts failed — build detailed diagnostics.
    _raise_parse_error(raw)


def _try_json_loads(text: str) -> Any:
    """Return parsed JSON or ``None`` on failure."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def _fix_literal_newlines(text: str) -> str:
    """Replace literal (unescaped) newlines inside JSON string values with ``\\n``.

    Walks the string tracking whether we are inside a JSON string (between
    unescaped double-quotes).  Any literal ``\\n`` or ``\\r`` encountered
    inside a string is replaced with the escaped form.
    """
    out: list[str] = []
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            if escaped:
                out.append(ch)
                escaped = False
                continue
            if ch == "\\":
                out.append(ch)
                escaped = True
                continue
            if ch == '"':
                in_string = False
                out.append(ch)
                continue
            if ch == "\n":
                out.append("\\n")
                continue
            if ch == "\r":
                out.append("\\r")
                continue
            out.append(ch)
        else:
            if ch == '"':
                in_string = True
            out.append(ch)
    return "".join(out)


def _raise_parse_error(raw: str) -> None:
    """Raise ``ValueError`` with rich diagnostics about the parse failure."""
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()

    diag_parts = ["LLM did not return valid JSON"]
    try:
        json.loads(text)
    except json.JSONDecodeError as exc:
        diag_parts.append(f"error_type={type(exc).__name__}")
        diag_parts.append(f"error_message={exc.msg}")
        diag_parts.append(f"line={exc.lineno}")
        diag_parts.append(f"column={exc.colno}")
        diag_parts.append(f"position={exc.pos}")
    except Exception:
        pass

    diag_parts.append(f"raw_response_preview={_preview(raw)}")
    diag_parts.append(f"raw_response_repr_preview={repr(raw[:1000])}")

    raise ValueError("; ".join(diag_parts))


def validate_answer_payload(
    parsed: dict[str, Any],
    *,
    answer_mode: str | None = None,
) -> dict[str, Any]:
    """Validate and normalise a parsed LLM answer payload.

    Parameters
    ----------
    parsed : dict
        The parsed JSON dict from the LLM.
    answer_mode : str, optional
        The current answer mode (``"grounded"``, ``"external_assisted"``, etc.).
        When set, schema requirements are adjusted per mode.

    Returns a dict with:
    - ``payload``: the normalised answer dict.
    - ``schema_error``: ``None`` if schema is fine, or a descriptive string
      if required fields were missing and had to be inferred.
    - ``warning``: any additional warning text, or ``None``.
    """
    final_answer = _string_value(parsed.get("final_answer"))
    answer_from_sources = _string_value(parsed.get("answer_from_sources"))
    warning = _string_value(parsed.get("warning"))

    # --- Mode-specific schema rules ---
    # For external_assisted / insufficient: answer_from_sources is NOT required.
    mode_allows_missing_sources = answer_mode in ("external_assisted", "insufficient")

    if final_answer and answer_from_sources:
        # Both present — ideal case.
        return {"payload": parsed, "schema_error": None, "warning": warning}

    if final_answer and not answer_from_sources:
        if mode_allows_missing_sources:
            # Expected for external_assisted — no schema error, no warning.
            normalised = dict(parsed)
            normalised["answer_from_sources"] = None
            return {"payload": normalised, "schema_error": None, "warning": warning}
        # Grounded/assisted — tolerate but flag as schema error.
        normalised = dict(parsed)
        normalised["answer_from_sources"] = final_answer
        return {
            "payload": normalised,
            "schema_error": "missing_required_fields: answer_from_sources",
            "warning": warning,
        }

    if answer_from_sources and not final_answer:
        normalised = dict(parsed)
        normalised["final_answer"] = answer_from_sources
        return {
            "payload": normalised,
            "schema_error": "missing_required_fields: final_answer",
            "warning": warning,
        }

    # Neither field is present — cannot salvage.
    return {
        "payload": parsed,
        "schema_error": "missing_required_fields: answer_from_sources, final_answer",
        "warning": warning,
    }


def _extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _string_value(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    value = value.strip()
    return value or None


def _merge_warning(left: str | None, right: str | None) -> str | None:
    values = [v for v in (left, right) if v]
    if not values:
        return None
    unique: list[str] = []
    for v in values:
        if v not in unique:
            unique.append(v)
    return " ".join(unique)

def _preview(value: str, *, limit: int = 500) -> str:
    compact = " ".join((value or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


# ---------------------------------------------------------------------------
# Light postprocessor for LLM-generated answer text
# ---------------------------------------------------------------------------

def clean_generated_text(text: str) -> str:
    """Light cleanup of LLM-generated Arabic answers.

    - Normalise repeated whitespace.
    - Trim excessive blank lines (collapse 3+ consecutive blank lines to 2).
    - Collapse obviously repeated Arabic letters (3+ identical consecutive).
    - Does NOT alter legal source quotes or citations.
    """
    if not text:
        return text

    # Normalise spaces (but keep newlines).
    result = re.sub(r"[^\S\n]+", " ", text)

    # Collapse 3+ consecutive blank lines to 2.
    result = re.sub(r"\n{4,}", "\n\n\n", result)

    # Collapse 3+ identical consecutive Arabic letters  (e.g. "ههههه" → "هه").
    result = re.sub(r"([\u0600-\u06FF])\1{2,}", r"\1\1", result)

    return result.strip()


# Mode-specific max_completion_tokens defaults.
MODE_MAX_TOKENS: dict[str, int] = {
    "external_assisted": 3072,
    "grounded": 6144,
    "assisted": 4096,
    "insufficient": 2048,
}
