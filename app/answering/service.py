from __future__ import annotations

import re
import time
from typing import Any

from app.answering.intent_router import IntentDecision, IntentType, route_intent, sanitize_optional_value
from app.answering.prompts import build_answer_messages
from app.answering.schemas import AnswerParts, LegalAnswerResponse, LLMCallMetadata, RetrievalSummary, RouterMetadata, SourceCitation, TimingMetadata
from app.answering.source_sufficiency import EvaluatedSource, assess_source_sufficiency
from app.llm import LLMError, MODE_MAX_TOKENS, OpenAICompatibleLLMClient, clean_generated_text, parse_llm_json, validate_answer_payload
from app.models import RetrievalFilters
from app.preprocessing import normalize_legal_arabic
from app.retrieval import LegalRetriever
from app.runtime_assets import ensure_runtime_assets
from app.settings import Settings, settings


CHAT_EXTERNAL_ASSISTED_WARNING = (
    "هذا السؤال خارج مصادر التطبيق الداخلية المتاحة، لذلك لا أستطيع توثيق الإجابة منها."
)

PUBLIC_CHAT_LLM_FALLBACK_WARNING = (
    "تعذر توليد الصياغة النهائية حاليًا، وتم عرض إجابة مستندة إلى المصادر الداخلية المتاحة."
)

PUBLIC_CHAT_LLM_UNAVAILABLE_WARNING = "تعذر توليد الصياغة النهائية حاليًا."

CHAT_CONCISE_MAX_TOKENS: dict[str, int] = {
    "grounded": 1536,
    "assisted": 1536,
    "external_assisted": 1536,
    "insufficient": 1024,
}


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
        fallback_llm_client: OpenAICompatibleLLMClient | None = None,
        config: Settings | None = None,
    ) -> None:
        self.settings = config or settings
        self._retriever = retriever
        self.llm_client = llm_client or OpenAICompatibleLLMClient(config=self.settings)
        self.fallback_llm_client = fallback_llm_client or _build_fallback_llm_client(self.settings)

    @property
    def retriever(self) -> LegalRetriever:
        if self._retriever is None:
            ensure_runtime_assets(config=self.settings)
            self._retriever = LegalRetriever(config=self.settings)
        return self._retriever

    def answer(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: RetrievalFilters | None = None,
        include_retrieval: bool = False,
        concise: bool = False,
    ) -> LegalAnswerResponse:
        if not query.strip():
            raise ValueError("query must not be empty")

        t_start = time.perf_counter()

        # --- Intent Router (runs before any retrieval) ---
        domain_hint = sanitize_optional_value(getattr(filters, "legal_domain", None)) if filters else None
        if domain_hint == "all":
            domain_hint = None
        intent = route_intent(query, explicit_domain=domain_hint)
        t_intent = time.perf_counter()

        if intent.intent == IntentType.IDENTITY:
            return _identity_response(query, intent, t_start, t_intent)

        if intent.intent == IntentType.CONVERSATION:
            return _conversation_response(query, intent, t_start, t_intent)

        if intent.intent == IntentType.NON_LEGAL:
            return _non_legal_response(query, intent, t_start, t_intent)

        if intent.intent == IntentType.EXTERNAL_ASSISTED:
            return self._external_assisted_shortcircuit(
                query,
                intent=intent,
                include_retrieval=include_retrieval,
                concise=concise,
                t_start=t_start,
                t_intent=t_intent,
            )

        if intent.intent == IntentType.AMBIGUOUS:
            return _ambiguous_response(query, intent, t_start, t_intent)

        # --- Legal retrieval path ---
        retrieval_filters = _effective_retrieval_filters(filters, suggested_domain=intent.suggested_domain)
        top_k_used = top_k or self.settings.legal_answer_top_k
        retrieval_result = self.retriever.search(query, top_k=top_k_used, filters=retrieval_filters)
        decision = assess_source_sufficiency(
            retrieval_result,
            config=self.settings,
            top_k=top_k_used,
            explicit_domain=domain_hint,
            has_legal_intent=intent.is_legal_question,
        )
        if _should_retry_broad_retrieval(retrieval_filters, decision):
            broad_filters = _broad_retrieval_filters(retrieval_filters)
            broad_retrieval_result = self.retriever.search(query, top_k=top_k_used, filters=broad_filters)
            broad_decision = assess_source_sufficiency(
                broad_retrieval_result,
                config=self.settings,
                top_k=top_k_used,
                explicit_domain=domain_hint,
                has_legal_intent=intent.is_legal_question,
            )
            if _broad_decision_is_better(decision, broad_decision):
                retrieval_result = broad_retrieval_result
                decision = broad_decision
                decision.reasons.append("broad_retrieval_retry_used")
        t_retrieval = time.perf_counter()

        internal_sources = [source.citation for source in decision.sources if decision.answer_mode in {"grounded", "assisted"}]
        internal_source_blocks = self._build_source_blocks(decision.sources if decision.answer_mode in {"grounded", "assisted"} else [])
        external_sources: list[SourceCitation] = []
        external_sources_verified_by_system = False

        # Mode-specific max_completion_tokens.
        mode_max_tokens = _mode_max_tokens(decision.answer_mode, concise=concise, config=self.settings)

        llm, llm_payload, llm_parse_warning = self._call_llm(
            query=query,
            decision=decision,
            internal_source_blocks=internal_source_blocks,
            max_tokens=mode_max_tokens,
            concise=concise,
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
            llm_error=None if concise else llm.error,
            concise=concise,
        )

        # Apply light cleanup to LLM-generated text (not source quotes).
        if final_answer and llm_payload:
            final_answer = clean_generated_text(final_answer)
        if answer_from_sources and llm_payload:
            answer_from_sources = clean_generated_text(answer_from_sources)
        answer_parts = _build_answer_parts(
            mode=decision.answer_mode,
            final_answer=final_answer,
            payload=llm_payload,
            internal_sources=internal_sources,
            external_sources=external_sources,
        )

        # --- Build user-facing warning (no technical text for public /chat) ---
        if concise and _llm_output_unusable(llm, llm_payload):
            warning = _public_chat_llm_warning(decision.answer_mode, has_internal_sources=bool(internal_sources))
        elif llm.error:
            warning = _merge_warning(warning, "تعذر استدعاء نموذج اللغة؛ تم إرجاع output آمن بدلًا من فشل الطلب.")
        if decision.answer_mode == "external_assisted":
            warning = _merge_warning(
                CHAT_EXTERNAL_ASSISTED_WARNING if concise else EXTERNAL_ASSISTED_WARNING,
                warning if concise and warning != CHAT_EXTERNAL_ASSISTED_WARNING else None,
            )

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
            answer_parts=answer_parts,
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
            timing=TimingMetadata(
                intent_ms=(t_intent - t_start) * 1000,
                retrieval_ms=(t_retrieval - t_intent) * 1000,
                llm_ms=(time.perf_counter() - t_retrieval) * 1000,
                total_ms=(time.perf_counter() - t_start) * 1000,
            ),
            router=_router_metadata(intent),
        )

    def _external_assisted_shortcircuit(
        self,
        query: str,
        *,
        intent: IntentDecision,
        include_retrieval: bool = False,
        concise: bool = False,
        t_start: float,
        t_intent: float,
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
        mode_max_tokens = _mode_max_tokens("external_assisted", concise=concise, config=self.settings)

        llm, llm_payload, llm_parse_warning = self._call_llm(
            query=query,
            decision=decision,
            internal_source_blocks=[],
            max_tokens=mode_max_tokens,
            concise=concise,
        )

        final_answer, answer_from_sources, external_or_assisted_explanation, warning = self._answer_fields(
            mode="external_assisted",
            payload=llm_payload,
            internal_sources=[],
            external_sources=[],
            llm_error=None if concise else llm.error,
            concise=concise,
        )

        if final_answer and llm_payload:
            final_answer = clean_generated_text(final_answer)
        answer_parts = _build_answer_parts(
            mode="external_assisted",
            final_answer=final_answer,
            payload=llm_payload,
            internal_sources=[],
            external_sources=[],
        )

        if concise and _llm_output_unusable(llm, llm_payload):
            warning = CHAT_EXTERNAL_ASSISTED_WARNING
        elif llm.error:
            warning = _merge_warning(warning, "تعذر استدعاء نموذج اللغة؛ تم إرجاع output آمن بدلًا من فشل الطلب.")
        else:
            warning = CHAT_EXTERNAL_ASSISTED_WARNING if concise else EXTERNAL_ASSISTED_WARNING

        return LegalAnswerResponse(
            query=query,
            answer_mode="external_assisted",
            is_out_of_internal_corpus=True,
            internal_grounding_sufficient=False,
            final_answer=final_answer,
            answer_parts=answer_parts,
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
            timing=TimingMetadata(
                intent_ms=(t_intent - t_start) * 1000,
                retrieval_ms=0,
                llm_ms=(time.perf_counter() - t_intent) * 1000,
                total_ms=(time.perf_counter() - t_start) * 1000,
            ),
            router=_router_metadata(intent),
        )

    def _call_llm(
        self,
        *,
        query: str,
        decision: Any,
        internal_source_blocks: list[dict[str, Any]],
        max_tokens: int,
        concise: bool = False,
    ) -> tuple[LLMCallMetadata, dict[str, Any] | None, str | None]:
        """Call the LLM and parse/validate the response. Returns (llm_meta, payload, parse_warning)."""
        llm_parse_warning: str | None = None
        self._last_completion_raw = None
        messages = build_answer_messages(
            query=query,
            answer_mode=decision.answer_mode,
            internal_grounding_sufficient=decision.internal_grounding_sufficient,
            is_out_of_internal_corpus=decision.is_out_of_internal_corpus,
            sufficiency_reasons=decision.reasons,
            internal_sources=internal_source_blocks,
            external_sources=[],
            external_sources_verified_by_system=False,
            concise=concise,
            detail_level=self.settings.chat_answer_detail_level,
        )

        primary_llm, primary_payload, primary_raw = self._call_one_llm(
            self.llm_client,
            messages=messages,
            max_tokens=max_tokens,
            answer_mode=decision.answer_mode,
        )
        primary_llm.primary_provider = getattr(self.llm_client, "provider_name", self.settings.llm_provider_name)
        primary_llm.primary_model = getattr(self.llm_client, "model", None)

        if not _llm_output_unusable(primary_llm, primary_payload):
            self._last_completion_raw = primary_raw
            return primary_llm, primary_payload, llm_parse_warning

        primary_error = _llm_attempt_error(primary_llm)
        fallback_client = self.fallback_llm_client
        if fallback_client is None:
            primary_llm.primary_error = primary_error
            return primary_llm, primary_payload, llm_parse_warning

        fallback_llm, fallback_payload, fallback_raw = self._call_one_llm(
            fallback_client,
            messages=messages,
            max_tokens=max_tokens,
            answer_mode=decision.answer_mode,
        )
        fallback_llm.primary_provider = primary_llm.primary_provider
        fallback_llm.primary_model = primary_llm.primary_model
        fallback_llm.primary_error = primary_error
        fallback_llm.fallback_provider = getattr(fallback_client, "provider_name", None)
        fallback_llm.fallback_model = getattr(fallback_client, "model", None)

        if not _llm_output_unusable(fallback_llm, fallback_payload):
            fallback_llm.fallback_used = True
            self._last_completion_raw = fallback_raw
            return fallback_llm, fallback_payload, llm_parse_warning

        primary_llm.succeeded = False
        primary_llm.primary_error = primary_error
        primary_llm.fallback_provider = fallback_llm.fallback_provider
        primary_llm.fallback_model = fallback_llm.fallback_model
        primary_llm.fallback_error = _llm_attempt_error(fallback_llm)
        return primary_llm, None, llm_parse_warning

    def _call_one_llm(
        self,
        client: OpenAICompatibleLLMClient,
        *,
        messages: list[dict[str, str]],
        max_tokens: int,
        answer_mode: str,
    ) -> tuple[LLMCallMetadata, dict[str, Any] | None, dict[str, Any] | None]:
        llm = LLMCallMetadata(
            provider=getattr(client, "provider_name", self.settings.llm_provider_name),
            model=getattr(client, "model", None),
            called=True,
            web_search_enabled=bool(getattr(client, "web_search_enabled", self.settings.llm_web_search_enabled)),
        )
        llm_payload: dict[str, Any] | None = None
        raw_response: dict[str, Any] | None = None

        try:
            completion = client.chat_completion(messages=messages, temperature=0.0, max_tokens=max_tokens)
            raw_response = completion.raw_response
            llm = LLMCallMetadata(
                provider=getattr(client, "provider_name", self.settings.llm_provider_name),
                model=completion.model,
                called=True,
                succeeded=True,
                usage=completion.usage,
                web_search_enabled=bool(getattr(client, "web_search_enabled", self.settings.llm_web_search_enabled)),
            )
            try:
                parsed = parse_llm_json(completion.content)
            except ValueError as exc:
                llm.parse_error = str(exc)
                llm.raw_response_preview = _preview(completion.content)
                llm.raw_response_repr_preview = repr(completion.content[:1000])
            else:
                validation = validate_answer_payload(parsed, answer_mode=answer_mode)
                llm_payload = validation["payload"]
                if validation.get("schema_error"):
                    llm.schema_error = validation["schema_error"]
        except LLMError as exc:
            llm.error = str(exc)
            llm.succeeded = False
        except Exception as exc:
            llm.error = f"Unexpected LLM error: {exc}"
            llm.succeeded = False

        return llm, llm_payload, raw_response

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
        concise: bool = False,
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
                    warning = _merge_warning(warning, CHAT_EXTERNAL_ASSISTED_WARNING if concise else EXTERNAL_ASSISTED_WARNING)
                    answer_from_sources = None
                return final_answer, answer_from_sources, external_or_assisted_explanation, warning

        return _fallback_answer(
            mode=mode,
            internal_sources=internal_sources,
            external_sources=external_sources,
            llm_error=llm_error,
            concise=concise,
        )


def _identity_response(query: str, intent: IntentDecision, t_start: float, t_intent: float) -> LegalAnswerResponse:
    t_total = time.perf_counter()
    return LegalAnswerResponse(
        query=query,
        answer_mode="identity",
        is_out_of_internal_corpus=False,
        internal_grounding_sufficient=False,
        final_answer=IDENTITY_ANSWER,
        answer_parts=_simple_answer_parts(IDENTITY_ANSWER),
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
        is_legal_question=False,
        is_supported_by_internal_sources=False,
        timing=TimingMetadata(
            intent_ms=(t_intent - t_start) * 1000,
            retrieval_ms=0,
            llm_ms=0,
            total_ms=(t_total - t_start) * 1000,
        ),
        router=_router_metadata(intent),
    )


def _conversation_response(query: str, intent: Any, t_start: float, t_intent: float) -> LegalAnswerResponse:
    """Local response for greetings / thanks / small-talk — no retrieval, no LLM."""
    t_total = time.perf_counter()
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
        answer_parts=_simple_answer_parts(answer),
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
        timing=TimingMetadata(
            intent_ms=(t_intent - t_start) * 1000,
            retrieval_ms=0,
            llm_ms=0,
            total_ms=(t_total - t_start) * 1000,
        ),
        router=_router_metadata(intent),
    )


def _non_legal_response(query: str, intent: Any, t_start: float, t_intent: float) -> LegalAnswerResponse:
    """Local response for non-legal queries — no retrieval, no LLM."""
    t_total = time.perf_counter()
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
        answer_parts=_simple_answer_parts(NON_LEGAL_ANSWER),
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
        timing=TimingMetadata(
            intent_ms=(t_intent - t_start) * 1000,
            retrieval_ms=0,
            llm_ms=0,
            total_ms=(t_total - t_start) * 1000,
        ),
        router=_router_metadata(intent),
    )


def _ambiguous_response(query: str, intent: Any, t_start: float, t_intent: float) -> LegalAnswerResponse:
    """Fast-path for ambiguous queries that need clarification (no LLM, no Qdrant)."""
    t_total = time.perf_counter()
    empty_summary = RetrievalSummary(
        domain=None, law=None, top_k_used=0, result_count=0,
        source_count=0, internal_source_count=0, external_source_count=0,
        sufficiency_reasons=["ambiguous_routed_without_retrieval"],
        sufficiency_metrics={"shortcircuit": True},
    )
    return LegalAnswerResponse(
        query=query,
        answer_mode="insufficient",
        is_out_of_internal_corpus=False,
        internal_grounding_sufficient=False,
        final_answer="من فضلك وضّح سؤالك القانوني أو اذكر المجال القانوني المطلوب.",
        answer_parts=_simple_answer_parts("من فضلك وضّح سؤالك القانوني أو اذكر المجال القانوني المطلوب."),
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
        is_legal_question=intent.is_legal_question,
        is_supported_by_internal_sources=False,
        timing=TimingMetadata(
            intent_ms=(t_intent - t_start) * 1000,
            retrieval_ms=0,
            llm_ms=0,
            total_ms=(t_total - t_start) * 1000,
        ),
        router=_router_metadata(intent),
    )


def _router_metadata(intent: IntentDecision) -> RouterMetadata:
    return RouterMetadata(
        intent=intent.intent.value,
        confidence=round(float(intent.confidence), 4),
        suggested_domain=intent.suggested_domain,
        is_legal_question=bool(intent.is_legal_question),
        is_out_of_internal_corpus=bool(intent.is_out_of_internal_corpus),
        reasons=list(intent.reasons),
        scores={key: round(float(value), 4) for key, value in intent.scores.items()},
    )


def _effective_retrieval_filters(
    filters: RetrievalFilters | None,
    *,
    suggested_domain: str | None,
) -> RetrievalFilters | None:
    legal_domain = sanitize_optional_value(getattr(filters, "legal_domain", None)) if filters else None
    if legal_domain == "all":
        legal_domain = None
    if not legal_domain:
        legal_domain = suggested_domain
    law_number = sanitize_optional_value(getattr(filters, "law_number", None)) if filters else None
    law_year = sanitize_optional_value(getattr(filters, "law_year", None)) if filters else None
    status_normalized = sanitize_optional_value(getattr(filters, "status_normalized", None)) if filters else None
    exclude_repealed = bool(getattr(filters, "exclude_repealed", False)) if filters else False
    if not any((legal_domain, law_number, law_year, status_normalized, exclude_repealed)):
        return None
    return RetrievalFilters(
        legal_domain=legal_domain,
        law_number=law_number,
        law_year=law_year,
        status_normalized=status_normalized,
        exclude_repealed=exclude_repealed,
    )


def _should_retry_broad_retrieval(
    filters: RetrievalFilters | None,
    decision: Any,
) -> bool:
    if not filters or not getattr(filters, "legal_domain", None):
        return False
    if decision.answer_mode in {"grounded", "assisted"}:
        return False
    metrics = getattr(decision, "metrics", {}) or {}
    return (
        metrics.get("usable_source_count", 0) == 0
        or metrics.get("partial_source_count", 0) == 0
        or decision.answer_mode == "insufficient"
    )


def _broad_retrieval_filters(filters: RetrievalFilters | None) -> RetrievalFilters | None:
    if not filters:
        return None
    if not any((
        filters.law_number,
        filters.law_year,
        filters.status_normalized,
        filters.exclude_repealed,
    )):
        return None
    return RetrievalFilters(
        legal_domain=None,
        law_number=filters.law_number,
        law_year=filters.law_year,
        status_normalized=filters.status_normalized,
        exclude_repealed=filters.exclude_repealed,
    )


def _broad_decision_is_better(current: Any, broad: Any) -> bool:
    rank = {"insufficient": 0, "external_assisted": 1, "assisted": 2, "grounded": 3}
    current_rank = rank.get(current.answer_mode, 0)
    broad_rank = rank.get(broad.answer_mode, 0)
    if broad_rank > current_rank:
        return True
    current_metrics = getattr(current, "metrics", {}) or {}
    broad_metrics = getattr(broad, "metrics", {}) or {}
    return (
        broad_rank == current_rank
        and broad_metrics.get("partial_source_count", 0) > current_metrics.get("partial_source_count", 0)
    )


def _mode_max_tokens(mode: str, *, concise: bool, config: Settings) -> int:
    full_budget = MODE_MAX_TOKENS.get(mode, config.llm_max_tokens)
    if not concise:
        return full_budget
    return min(full_budget, CHAT_CONCISE_MAX_TOKENS.get(mode, 1536))


def _build_fallback_llm_client(config: Settings) -> OpenAICompatibleLLMClient | None:
    if not config.llm_fallback_provider_name or not config.llm_fallback_api_key:
        return None
    return OpenAICompatibleLLMClient(
        api_key=config.llm_fallback_api_key,
        base_url=config.llm_fallback_base_url,
        model=config.llm_fallback_model,
        provider_name=config.llm_fallback_provider_name,
        timeout_seconds=config.llm_timeout_seconds,
        config=config,
    )


def _simple_answer_parts(final_answer: str) -> AnswerParts:
    return AnswerParts(
        intro=final_answer,
        section_title=None,
        bullets=[],
        legal_basis=None,
        note=None,
    )


def _build_answer_parts(
    *,
    mode: str,
    final_answer: str,
    payload: dict[str, Any] | None,
    internal_sources: list[SourceCitation],
    external_sources: list[SourceCitation],
) -> AnswerParts:
    from_payload = _answer_parts_from_payload(payload)
    if from_payload is not None:
        if mode in {"grounded", "assisted"}:
            source_basis = _legal_basis_from_sources(internal_sources)
            if source_basis:
                from_payload.legal_basis = source_basis
            elif not from_payload.legal_basis:
                from_payload.legal_basis = None
        if mode == "external_assisted":
            from_payload.legal_basis = from_payload.legal_basis if external_sources else None
            from_payload.note = from_payload.note or "هذه إجابة عامة وليست مستندة إلى مصادر داخلية موثقة."
        return from_payload

    if mode in {"identity", "conversation", "non_legal", "insufficient"}:
        return _simple_answer_parts(final_answer)

    parsed = _answer_parts_from_final_answer(final_answer)
    if mode in {"grounded", "assisted"}:
        parsed.legal_basis = parsed.legal_basis or _legal_basis_from_sources(internal_sources)
    elif mode == "external_assisted":
        parsed.section_title = parsed.section_title or "شرح عام:"
        parsed.legal_basis = None
        parsed.note = parsed.note or "هذه إجابة عامة وليست مستندة إلى مصادر داخلية موثقة."
    return parsed


def _answer_parts_from_payload(payload: dict[str, Any] | None) -> AnswerParts | None:
    if not payload or not isinstance(payload.get("answer_parts"), dict):
        return None
    raw = payload["answer_parts"]
    bullets = raw.get("bullets")
    clean_bullets = [
        bullet.strip().lstrip("-• ").strip()
        for bullet in bullets
        if isinstance(bullet, str) and bullet.strip()
    ] if isinstance(bullets, list) else []
    return AnswerParts(
        intro=_string_value(raw.get("intro")),
        section_title=_normalise_heading(_string_value(raw.get("section_title"))),
        bullets=clean_bullets[:6],
        legal_basis=_string_value(raw.get("legal_basis")),
        note=_string_value(raw.get("note")),
    )


def _answer_parts_from_final_answer(final_answer: str) -> AnswerParts:
    lines = [line.strip() for line in (final_answer or "").splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return _simple_answer_parts(final_answer)

    section_index = _first_heading_index(lines)
    intro = " ".join(lines[:section_index]).strip() if section_index is not None and section_index > 0 else lines[0]
    section_title = _normalise_heading(lines[section_index]) if section_index is not None else None

    bullets: list[str] = []
    legal_basis_lines: list[str] = []
    note_lines: list[str] = []
    active: str | None = None

    for line in lines[(section_index + 1 if section_index is not None else 1):]:
        normalised = line.rstrip(":").strip()
        if _is_answer_bullet_line(line):
            bullet = _clean_answer_bullet_line(line)
            if bullet:
                bullets.append(bullet)
            continue
        if normalised in {"السند القانوني", "سند قانوني"}:
            active = "legal_basis"
            continue
        if normalised in {"ملاحظة", "تنبيه"}:
            active = "note"
            continue
        if _looks_like_section_heading(line) and not section_title:
            section_title = _normalise_heading(line)
            continue
        if active == "legal_basis":
            legal_basis_lines.append(line)
        elif active == "note":
            note_lines.append(line)

    return AnswerParts(
        intro=intro or final_answer,
        section_title=section_title,
        bullets=bullets[:6],
        legal_basis=" ".join(legal_basis_lines).strip() or None,
        note=" ".join(note_lines).strip() or None,
    )


def _first_heading_index(lines: list[str]) -> int | None:
    for index, line in enumerate(lines):
        if _looks_like_section_heading(line):
            return index
    return None


def _looks_like_section_heading(line: str) -> bool:
    text = line.rstrip(":").strip()
    return text in {
        "أهم الأحكام",
        "أهم الضمانات",
        "أبرز الحقوق والضمانات",
        "الخطوات العملية",
        "ما يمكنك فعله",
        "شرح عام",
    }


def _normalise_heading(value: str | None) -> str | None:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    return text if text.endswith(":") else f"{text}:"


def _is_answer_bullet_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith(("-", "•", "*")) or bool(re.match(r"^[0-9٠-٩١٢٣٤٥٦٧٨٩]+[.)-]\s+", stripped))


def _clean_answer_bullet_line(line: str) -> str:
    stripped = line.strip().lstrip("-•* ").strip()
    return re.sub(r"^[0-9٠-٩١٢٣٤٥٦٧٨٩]+[.)-]\s+", "", stripped).strip()


def _legal_basis_from_sources(sources: list[SourceCitation]) -> str | None:
    entries: list[str] = []
    for source in sources:
        article = source.article_number
        law = source.law_name or source.document_level
        if article and law:
            entry = f"المادة {article} من {law}"
        elif article:
            entry = f"المادة {article}"
        elif law:
            entry = law
        else:
            continue
        if entry not in entries:
            entries.append(entry)
        if len(entries) >= 3:
            break
    if not entries:
        return None
    return "استندت الإجابة إلى " + " و".join(entries) + "."


def _fallback_answer(
    *,
    mode: str,
    internal_sources: list[SourceCitation],
    external_sources: list[SourceCitation],
    llm_error: str | None,
    concise: bool = False,
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
        if concise:
            explanation = (
                "شرح عام:\n"
                "- يمكن تقديم تصور عام للمسألة في سياق القانون المصري دون اعتباره توثيقًا داخليًا.\n"
                "- تختلف التفاصيل حسب الوقائع والنصوص الرسمية المطبقة.\n"
                "- لا توجد مصادر داخلية معتمدة لهذا الموضوع في نتيجة التطبيق الحالية.\n\n"
                "ملاحظة:\n"
                "هذه إجابة عامة غير موثقة من مصادر التطبيق الداخلية، ويُفضّل مراجعة محامٍ مختص أو النصوص الرسمية."
            )
            final_answer = f"{CHAT_EXTERNAL_ASSISTED_WARNING}\n\n{explanation}"
            return final_answer, None, explanation, _merge_warning(CHAT_EXTERNAL_ASSISTED_WARNING, _llm_error_warning(llm_error))
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


def _llm_output_unusable(llm: LLMCallMetadata, payload: dict[str, Any] | None) -> bool:
    if llm.error or llm.parse_error or llm.schema_error:
        return True
    return bool(llm.called and llm.succeeded and payload is None)


def _llm_attempt_error(llm: LLMCallMetadata) -> str | None:
    return llm.error or llm.parse_error or llm.schema_error


def _public_chat_llm_warning(mode: str, *, has_internal_sources: bool) -> str:
    if mode in {"grounded", "assisted"} and has_internal_sources:
        return PUBLIC_CHAT_LLM_FALLBACK_WARNING
    if mode == "external_assisted":
        return CHAT_EXTERNAL_ASSISTED_WARNING
    return PUBLIC_CHAT_LLM_UNAVAILABLE_WARNING


def _merge_warning(left: str | None, right: str | None) -> str | None:
    values = [value for value in (left, right) if value]
    if not values:
        return None
    unique: list[str] = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return " ".join(unique)
