from __future__ import annotations

from typing import Any

from app.answering.intent_router import IntentType, route_intent
from app.answering.prompts import build_answer_messages
from app.answering.schemas import LegalAnswerResponse, LLMCallMetadata, RetrievalSummary, SourceCitation
from app.answering.source_sufficiency import EvaluatedSource, assess_source_sufficiency
from app.llm import LLMError, MODE_MAX_TOKENS, OpenAICompatibleLLMClient, clean_generated_text, parse_llm_json, validate_answer_payload
from app.models import RetrievalFilters
from app.preprocessing import normalize_legal_arabic
from app.retrieval import LegalRetriever
from app.settings import Settings, settings


IDENTITY_ANSWER = (
    "أنا المستشار، مساعد قانوني ذكي ضمن تطبيق المستشار، تم تصميمي وتطويري بواسطة لؤي وائل. "
    "أساعدك في فهم وشرح المسائل القانونية في نطاق القانون المصري، مع توضيح ما إذا كانت الإجابة "
    "مبنية على مصادر التطبيق الداخلية أو على شرح مساعد."
)

EXTERNAL_ASSISTED_WARNING = (
    "هذا السؤال خارج قاعدة المصادر الداخلية الحالية، لذلك لا يمكن توثيق الإجابة من مواد النظام، "
    "ويمكن تقديم شرح عام مساعد فقط."
)

CONVERSATION_GENERIC = "أهلًا بك، أنا المستشار. اكتب سؤالك القانوني المصري وسأحاول مساعدتك."

CONVERSATION_RESPONSES: dict[str, str] = {
    "greeting": "وعليكم السلام، أهلًا بك. أنا المستشار، كيف يمكنني مساعدتك في سؤال قانوني مصري؟",
    "thanks": "العفو، يسعدني مساعدتك في أي سؤال قانوني مصري.",
}

NON_LEGAL_ANSWER = (
    "أنا المستشار، مساعد قانوني مخصص للأسئلة المتعلقة بالقانون المصري. "
    "من فضلك اكتب سؤالًا قانونيًا لأتمكن من مساعدتك."
)

AMBIGUOUS_ANSWER = "من فضلك وضّح سؤالك القانوني أو اذكر المجال القانوني المطلوب."


class LegalAnswerService:
    def __init__(
        self,
        *,
        retriever: LegalRetriever | None = None,
        llm_client: OpenAICompatibleLLMClient | None = None,
        config: Settings | None = None,
    ) -> None:
        self.settings = config or settings
        self._retriever = retriever
        self.llm_client = llm_client or OpenAICompatibleLLMClient(config=self.settings)

    @property
    def retriever(self) -> LegalRetriever:
        if self._retriever is None:
            self._retriever = LegalRetriever(config=self.settings)
        return self._retriever

    def answer(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: RetrievalFilters | None = None,
        include_retrieval: bool = False,
    ) -> LegalAnswerResponse:
        if not query.strip():
            raise ValueError("query must not be empty")

        # --- Intent Router (runs before any retrieval) ---
        domain_hint = getattr(filters, "legal_domain", None) if filters else None
        intent = route_intent(query, explicit_domain=domain_hint)

        if intent.intent == IntentType.IDENTITY:
            return _identity_response(query)

        if intent.intent == IntentType.CONVERSATION:
            return _conversation_response(query, intent)

        if intent.intent == IntentType.NON_LEGAL:
            return _non_legal_response(query, intent)

        if intent.intent == IntentType.EXTERNAL_ASSISTED:
            return self._external_assisted_shortcircuit(query, include_retrieval=include_retrieval)

        # --- Legal retrieval path (or ambiguous fallback) ---
        top_k_used = top_k or self.settings.legal_answer_top_k
        retrieval_result = self.retriever.search(query, top_k=top_k_used, filters=filters)
        decision = assess_source_sufficiency(
            retrieval_result,
            config=self.settings,
            top_k=top_k_used,
            explicit_domain=domain_hint,
            has_legal_intent=intent.is_legal_question,
        )

        internal_sources = [source.citation for source in decision.sources if decision.answer_mode in {"grounded", "assisted"}]
        internal_source_blocks = self._build_source_blocks(decision.sources if decision.answer_mode in {"grounded", "assisted"} else [])
        external_sources: list[SourceCitation] = []
        external_sources_verified_by_system = False

        # Mode-specific max_completion_tokens.
        mode_max_tokens = MODE_MAX_TOKENS.get(decision.answer_mode, self.settings.llm_max_tokens)

        llm, llm_payload, llm_parse_warning = self._call_llm(
            query=query,
            decision=decision,
            internal_source_blocks=internal_source_blocks,
            max_tokens=mode_max_tokens,
        )

        if llm.succeeded and llm.raw_response_preview is None:
            # Extract external sources from raw LLM response only on success.
            completion_raw = getattr(self, "_last_completion_raw", None)
            if completion_raw:
                external_sources = _extract_external_sources(
                    completion_raw,
                    verified_by_system=bool(getattr(self.llm_client, "web_search_enabled", False)),
                )
                external_sources_verified_by_system = bool(external_sources) and all(
                    source.verified_by_system for source in external_sources
                )

        final_answer, answer_from_sources, external_or_assisted_explanation, warning = self._answer_fields(
            mode=decision.answer_mode,
            payload=llm_payload,
            internal_sources=internal_sources,
            external_sources=external_sources,
            llm_error=llm.error,
        )

        # Apply light cleanup to LLM-generated text (not source quotes).
        if final_answer and llm_payload:
            final_answer = clean_generated_text(final_answer)
        if answer_from_sources and llm_payload:
            answer_from_sources = clean_generated_text(answer_from_sources)

        # --- Build user-facing warning (no technical text) ---
        if llm.error:
            warning = _merge_warning(warning, "تعذر استدعاء نموذج اللغة؛ تم إرجاع output آمن بدلًا من فشل الطلب.")
        if decision.answer_mode == "external_assisted":
            warning = EXTERNAL_ASSISTED_WARNING

        # Semantic classification.
        is_legal = decision.answer_mode != "identity"
        is_out_of_domain_legacy = False  # Egyptian legal questions are NOT out-of-domain.
        if decision.is_out_of_internal_corpus and is_legal:
            is_out_of_domain_legacy = False  # Still an Egyptian law question.

        return LegalAnswerResponse(
            query=query,
            answer_mode=decision.answer_mode,
            is_out_of_internal_corpus=decision.is_out_of_internal_corpus,
            internal_grounding_sufficient=decision.internal_grounding_sufficient,
            final_answer=final_answer,
            answer_from_sources=answer_from_sources,
            external_or_assisted_explanation=external_or_assisted_explanation,
            warning=warning,
            internal_sources=internal_sources,
            external_sources=external_sources,
            external_sources_verified_by_system=external_sources_verified_by_system,
            retrieval_summary=RetrievalSummary(
                domain=decision.domain,
                law=decision.law,
                top_k_used=top_k_used,
                result_count=len(retrieval_result.get("results") or []),
                source_count=len(internal_sources),
                internal_source_count=len(internal_sources),
                external_source_count=len(external_sources),
                sufficiency_reasons=decision.reasons,
                sufficiency_metrics=decision.metrics,
            ),
            llm=llm,
            retrieval_result=retrieval_result if include_retrieval else None,
            is_out_of_domain=is_out_of_domain_legacy,
            grounding_sufficient=decision.internal_grounding_sufficient,
            assisted_explanation=external_or_assisted_explanation,
            sources=internal_sources,
            is_legal_question=is_legal,
            is_supported_by_internal_sources=decision.internal_grounding_sufficient,
        )

    def _external_assisted_shortcircuit(
        self,
        query: str,
        *,
        include_retrieval: bool = False,
    ) -> LegalAnswerResponse:
        """Fast path for queries confidently identified as out-of-internal-corpus."""
        from app.answering.source_sufficiency import SourceSufficiencyDecision

        decision = SourceSufficiencyDecision(
            answer_mode="external_assisted",
            internal_grounding_sufficient=False,
            is_out_of_internal_corpus=True,
            sources=[],
            reasons=["confident_out_of_internal_corpus_shortcircuit"],
            metrics={"shortcircuit": True},
            domain=None,
            law=None,
        )
        mode_max_tokens = MODE_MAX_TOKENS.get("external_assisted", 3072)

        llm, llm_payload, llm_parse_warning = self._call_llm(
            query=query,
            decision=decision,
            internal_source_blocks=[],
            max_tokens=mode_max_tokens,
        )

        final_answer, answer_from_sources, external_or_assisted_explanation, warning = self._answer_fields(
            mode="external_assisted",
            payload=llm_payload,
            internal_sources=[],
            external_sources=[],
            llm_error=llm.error,
        )

        if final_answer and llm_payload:
            final_answer = clean_generated_text(final_answer)

        if llm.error:
            warning = _merge_warning(warning, "تعذر استدعاء نموذج اللغة؛ تم إرجاع output آمن بدلًا من فشل الطلب.")
        else:
            warning = EXTERNAL_ASSISTED_WARNING

        return LegalAnswerResponse(
            query=query,
            answer_mode="external_assisted",
            is_out_of_internal_corpus=True,
            internal_grounding_sufficient=False,
            final_answer=final_answer,
            answer_from_sources=answer_from_sources,
            external_or_assisted_explanation=external_or_assisted_explanation,
            warning=warning,
            internal_sources=[],
            external_sources=[],
            external_sources_verified_by_system=False,
            retrieval_summary=RetrievalSummary(
                domain=None,
                law=None,
                top_k_used=0,
                result_count=0,
                source_count=0,
                internal_source_count=0,
                external_source_count=0,
                sufficiency_reasons=decision.reasons,
                sufficiency_metrics=decision.metrics,
            ),
            llm=llm,
            retrieval_result=None,
            is_out_of_domain=False,
            grounding_sufficient=False,
            assisted_explanation=external_or_assisted_explanation,
            sources=[],
            is_legal_question=True,
            is_supported_by_internal_sources=False,
        )

    def _call_llm(
        self,
        *,
        query: str,
        decision: Any,
        internal_source_blocks: list[dict[str, Any]],
        max_tokens: int,
    ) -> tuple[LLMCallMetadata, dict[str, Any] | None, str | None]:
        """Call the LLM and parse/validate the response. Returns (llm_meta, payload, parse_warning)."""
        llm = LLMCallMetadata(
            provider=getattr(self.llm_client, "provider_name", self.settings.llm_provider_name),
            model=getattr(self.llm_client, "model", None),
            called=True,
            web_search_enabled=bool(getattr(self.llm_client, "web_search_enabled", self.settings.llm_web_search_enabled)),
        )
        llm_payload: dict[str, Any] | None = None
        llm_parse_warning: str | None = None
        self._last_completion_raw = None

        try:
            completion = self.llm_client.chat_completion(
                messages=build_answer_messages(
                    query=query,
                    answer_mode=decision.answer_mode,
                    internal_grounding_sufficient=decision.internal_grounding_sufficient,
                    is_out_of_internal_corpus=decision.is_out_of_internal_corpus,
                    sufficiency_reasons=decision.reasons,
                    internal_sources=internal_source_blocks,
                    external_sources=[],
                    external_sources_verified_by_system=False,
                ),
                temperature=0.0,
                max_tokens=max_tokens,
            )
            self._last_completion_raw = completion.raw_response
            llm = LLMCallMetadata(
                provider=getattr(self.llm_client, "provider_name", self.settings.llm_provider_name),
                model=completion.model,
                called=True,
                succeeded=True,
                usage=completion.usage,
                web_search_enabled=bool(getattr(self.llm_client, "web_search_enabled", self.settings.llm_web_search_enabled)),
            )
            try:
                parsed = parse_llm_json(completion.content)
            except ValueError as exc:
                llm.parse_error = str(exc)
                llm.raw_response_preview = _preview(completion.content)
                llm.raw_response_repr_preview = repr(completion.content[:1000])
            else:
                validation = validate_answer_payload(parsed, answer_mode=decision.answer_mode)
                llm_payload = validation["payload"]
                if validation.get("schema_error"):
                    llm.schema_error = validation["schema_error"]
        except LLMError as exc:
            llm = LLMCallMetadata(
                provider=getattr(self.llm_client, "provider_name", self.settings.llm_provider_name),
                model=getattr(self.llm_client, "model", None),
                called=True,
                succeeded=False,
                error=str(exc),
                web_search_enabled=bool(getattr(self.llm_client, "web_search_enabled", self.settings.llm_web_search_enabled)),
            )
        except Exception as exc:
            llm = LLMCallMetadata(
                provider=getattr(self.llm_client, "provider_name", self.settings.llm_provider_name),
                model=getattr(self.llm_client, "model", None),
                called=True,
                succeeded=False,
                error=f"Unexpected LLM error: {exc}",
                web_search_enabled=bool(getattr(self.llm_client, "web_search_enabled", self.settings.llm_web_search_enabled)),
            )

        return llm, llm_payload, llm_parse_warning

    def _build_source_blocks(self, sources: list[EvaluatedSource]) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        total_chars = 0
        for index, source in enumerate(sources, start=1):
            text = _source_context_text(source.raw, limit=self.settings.legal_answer_source_char_limit)
            total_chars += len(text)
            if total_chars > self.settings.legal_answer_context_char_limit:
                break
            citation = source.citation.model_dump()
            blocks.append(
                {
                    "source_id": f"S{index}",
                    "score": round(source.score, 6),
                    "overlap": round(source.overlap, 6),
                    "metadata": citation,
                    "text": text,
                }
            )
        return blocks

    def _answer_fields(
        self,
        *,
        mode: str,
        payload: dict[str, Any] | None,
        internal_sources: list[SourceCitation],
        external_sources: list[SourceCitation],
        llm_error: str | None,
    ) -> tuple[str, str | None, str | None, str | None]:
        if payload:
            final_answer = _string_value(payload.get("final_answer"))
            answer_from_sources = _string_value(payload.get("answer_from_sources"))
            external_or_assisted_explanation = _string_value(
                payload.get("external_or_assisted_explanation")
                or payload.get("assisted_explanation")
                or payload.get("external_explanation")
            )
            warning = _string_value(payload.get("warning"))
            if final_answer:
                if mode == "external_assisted":
                    warning = _merge_warning(warning, EXTERNAL_ASSISTED_WARNING)
                    answer_from_sources = None
                return final_answer, answer_from_sources, external_or_assisted_explanation, warning

        return _fallback_answer(
            mode=mode,
            internal_sources=internal_sources,
            external_sources=external_sources,
            llm_error=llm_error,
        )


def _identity_response(query: str) -> LegalAnswerResponse:
    return LegalAnswerResponse(
        query=query,
        answer_mode="identity",
        is_out_of_internal_corpus=False,
        internal_grounding_sufficient=False,
        final_answer=IDENTITY_ANSWER,
        answer_from_sources=None,
        external_or_assisted_explanation=IDENTITY_ANSWER,
        warning=None,
        internal_sources=[],
        external_sources=[],
        external_sources_verified_by_system=False,
        retrieval_summary=RetrievalSummary(
            domain=None,
            law=None,
            top_k_used=0,
            result_count=0,
            source_count=0,
            internal_source_count=0,
            external_source_count=0,
            sufficiency_reasons=["identity_query"],
            sufficiency_metrics={"identity_routed_without_retrieval": True},
        ),
        llm=LLMCallMetadata(called=False, succeeded=False),
        is_out_of_domain=False,
        grounding_sufficient=False,
        assisted_explanation=IDENTITY_ANSWER,
        sources=[],
    )


def _conversation_response(query: str, intent: Any) -> LegalAnswerResponse:
    """Local response for greetings / thanks / small-talk — no retrieval, no LLM."""
    norm = normalize_legal_arabic(query)
    # Pick sub-category response
    answer = CONVERSATION_GENERIC
    for cue in ("شكر", "تسلم", "متشكر", "جزاك"):
        if cue in norm:
            answer = CONVERSATION_RESPONSES["thanks"]
            break
    else:
        for cue in ("سلام", "اهلا", "أهلا", "مرحب", "صباح", "مساء", "هاي", "ازيك", "إزيك"):
            if cue in norm:
                answer = CONVERSATION_RESPONSES["greeting"]
                break

    empty_summary = RetrievalSummary(
        domain=None, law=None, top_k_used=0, result_count=0,
        source_count=0, internal_source_count=0, external_source_count=0,
        sufficiency_reasons=["conversation_routed_without_retrieval"],
        sufficiency_metrics={"shortcircuit": True},
    )
    return LegalAnswerResponse(
        query=query,
        answer_mode="conversation",
        is_out_of_internal_corpus=False,
        internal_grounding_sufficient=False,
        final_answer=answer,
        answer_from_sources=None,
        external_or_assisted_explanation=None,
        warning=None,
        internal_sources=[],
        external_sources=[],
        external_sources_verified_by_system=False,
        retrieval_summary=empty_summary,
        llm=LLMCallMetadata(called=False, succeeded=False),
        is_out_of_domain=False,
        grounding_sufficient=False,
        assisted_explanation=None,
        sources=[],
        is_legal_question=False,
        is_supported_by_internal_sources=False,
    )


def _non_legal_response(query: str, intent: Any) -> LegalAnswerResponse:
    """Local response for non-legal queries — no retrieval, no LLM."""
    empty_summary = RetrievalSummary(
        domain=None, law=None, top_k_used=0, result_count=0,
        source_count=0, internal_source_count=0, external_source_count=0,
        sufficiency_reasons=["non_legal_routed_without_retrieval"],
        sufficiency_metrics={"shortcircuit": True},
    )
    return LegalAnswerResponse(
        query=query,
        answer_mode="non_legal",
        is_out_of_internal_corpus=False,
        internal_grounding_sufficient=False,
        final_answer=NON_LEGAL_ANSWER,
        answer_from_sources=None,
        external_or_assisted_explanation=None,
        warning=None,
        internal_sources=[],
        external_sources=[],
        external_sources_verified_by_system=False,
        retrieval_summary=empty_summary,
        llm=LLMCallMetadata(called=False, succeeded=False),
        is_out_of_domain=False,
        grounding_sufficient=False,
        assisted_explanation=None,
        sources=[],
        is_legal_question=False,
        is_supported_by_internal_sources=False,
    )

def _fallback_answer(
    *,
    mode: str,
    internal_sources: list[SourceCitation],
    external_sources: list[SourceCitation],
    llm_error: str | None,
) -> tuple[str, str | None, str | None, str | None]:
    internal_summary = _sources_summary(internal_sources)
    if mode == "grounded":
        answer_from_sources = (
            "تعذر توليد صياغة نهائية عبر نموذج اللغة. المصادر الداخلية المسترجعة التي يمكن مراجعتها هي:\n"
            f"{internal_summary}"
        )
        final_answer = (
            "تعذر توليد جواب مباشر عبر نموذج اللغة.\n\n"
            "السند القانوني\n"
            f"{internal_summary}\n\n"
            "المصادر\n"
            f"{_source_list(internal_sources)}"
        )
        return final_answer, answer_from_sources, None, _llm_error_warning(llm_error)

    if mode == "assisted":
        answer_from_sources = f"المصادر الداخلية المتاحة جزئية، وأبرزها:\n{internal_summary}" if internal_sources else None
        explanation = "لم يتم توليد شرح مساعد لأن نداء نموذج اللغة لم يكتمل."
        final_answer = (
            "ما ورد في المصادر الداخلية المتاحة\n"
            f"{answer_from_sources or 'لا توجد مصادر داخلية كافية صالحة للاقتباس.'}\n\n"
            "شرح مساعد\n"
            f"{explanation}\n\n"
            "تنبيه\n"
            "هذا الإخراج لا يتضمن نتيجة قانونية مؤكدة خارج المصادر الداخلية المسترجعة."
        )
        return final_answer, answer_from_sources, explanation, _llm_error_warning(llm_error)

    if mode == "external_assisted":
        external_summary = _sources_summary(external_sources)
        explanation = (
            "يمكن تقديم شرح عام مساعد في سياق القانون المصري، لكن لا توجد مصادر داخلية كافية لتوثيق "
            "الإجابة من corpus التطبيق. لا ينبغي اعتبار هذا الشرح سندًا قانونيًا موثقًا من النظام."
        )
        if external_sources:
            explanation += f"\n\nمصادر خارجية رصدها النظام:\n{external_summary}"
        final_answer = (
            f"{EXTERNAL_ASSISTED_WARNING}\n\n"
            "شرح عام مساعد\n"
            f"{explanation}"
        )
        return final_answer, None, explanation, _merge_warning(EXTERNAL_ASSISTED_WARNING, _llm_error_warning(llm_error))

    warning = "المصادر الحالية غير كافية ولا يوجد أساس كافٍ داخل النظام لتقديم إجابة قانونية مصرية موثقة."
    return warning, None, None, _merge_warning(warning, _llm_error_warning(llm_error))


def _extract_external_sources(raw_response: dict[str, Any] | None, *, verified_by_system: bool) -> list[SourceCitation]:
    if not raw_response:
        return []

    candidates: list[Any] = []
    if isinstance(raw_response.get("citations"), list):
        candidates.extend(raw_response["citations"])

    choices = raw_response.get("choices")
    if isinstance(choices, list) and choices:
        message = (choices[0] or {}).get("message") or {}
        for key in ("citations", "annotations"):
            values = message.get(key)
            if isinstance(values, list):
                candidates.extend(values)

    sources: list[SourceCitation] = []
    for item in candidates:
        source = _external_source_from_item(item, verified_by_system=verified_by_system)
        if source and not any(existing.source_url == source.source_url and existing.title == source.title for existing in sources):
            sources.append(source)
    return sources


def _external_source_from_item(item: Any, *, verified_by_system: bool) -> SourceCitation | None:
    if not isinstance(item, dict):
        return None
    nested = item.get("url_citation") if isinstance(item.get("url_citation"), dict) else {}
    title = item.get("title") or nested.get("title") or item.get("name")
    url = item.get("url") or item.get("source_url") or nested.get("url")
    snippet = item.get("snippet") or item.get("text") or nested.get("snippet")
    if not title and not url:
        return None
    return SourceCitation(
        source_type="external",
        verified_by_system=verified_by_system,
        title=_string_value(title),
        source_url=_string_value(url),
        summary_snippet=_string_value(snippet),
    )


def _source_context_text(item: dict[str, Any], *, limit: int) -> str:
    parts = [
        item.get("summary") or "",
        item.get("content") or "",
        _supporting_chunk_text(item.get("supporting_chunks") or []),
    ]
    text = "\n\n".join(part for part in parts if part).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _supporting_chunk_text(chunks: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        "\n".join(part for part in (chunk.get("title") or "", chunk.get("content") or "") if part)
        for chunk in chunks
    ).strip()


def _sources_summary(sources: list[SourceCitation]) -> str:
    if not sources:
        return "- لا توجد مصادر صالحة."
    lines = []
    for index, source in enumerate(sources, start=1):
        label = _source_label(source)
        snippet = source.quote_snippet or source.summary_snippet
        if snippet:
            lines.append(f"- [{index}] {label}: {snippet}")
        else:
            lines.append(f"- [{index}] {label}")
    return "\n".join(lines)


def _source_list(sources: list[SourceCitation]) -> str:
    if not sources:
        return "- لا توجد مصادر."
    return "\n".join(f"- [{index}] {_source_label(source)}" for index, source in enumerate(sources, start=1))


def _source_label(source: SourceCitation) -> str:
    parts = []
    if source.law_name:
        parts.append(source.law_name)
    if source.article_number:
        parts.append(f"المادة {source.article_number}")
    if source.title and source.title not in parts:
        parts.append(source.title)
    if source.source_url:
        parts.append(source.source_url)
    return " | ".join(parts) if parts else (source.id or "مصدر غير مسمى")


def _is_identity_query(query: str) -> bool:
    raw = query.strip().lower()
    normalized = normalize_legal_arabic(query)
    cues = (
        "اسمك",
        "اسمك ايه",
        "اسمك إيه",
        "انت مين",
        "أنت مين",
        "مين انت",
        "مين أنت",
        "من انت",
        "من أنت",
        "مين طورك",
        "من طورك",
        "مين صممك",
        "من صممك",
        "مين عملك",
        "من عملك",
        "انت تابع لتطبيق ايه",
        "أنت تابع لتطبيق إيه",
        "تطبيق ايه",
        "تطبيق إيه",
    )
    if any(normalize_legal_arabic(cue) in normalized for cue in cues):
        return True
    english_cues = ("who are you", "what is your name", "your name", "who developed you", "who built you")
    return any(cue in raw for cue in english_cues)


def _string_value(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    value = value.strip()
    return value or None


def _preview(value: str, *, limit: int = 500) -> str:
    compact = " ".join((value or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _llm_error_warning(error: str | None) -> str | None:
    if not error:
        return None
    return f"خطأ نموذج اللغة: {error}"


def _merge_warning(left: str | None, right: str | None) -> str | None:
    values = [value for value in (left, right) if value]
    if not values:
        return None
    unique: list[str] = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return " ".join(unique)
