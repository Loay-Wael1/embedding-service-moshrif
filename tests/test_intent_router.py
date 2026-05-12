"""Tests for intent router fixes and ambiguous-fallthrough behavior."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from app.answering.intent_router import IntentType, route_intent


# ---------------------------------------------------------------------------
# Router unit tests — taa-marbuta normalization
# ---------------------------------------------------------------------------

class TestTaaMarbutaNormalization:
    """ة vs ه variants must route identically."""

    def test_formal_taa_marbuta_routes_to_legal(self):
        decision = route_intent("ما عقوبة السرقة")
        assert decision.intent == IntentType.LEGAL_RETRIEVAL
        assert decision.is_legal_question is True
        assert decision.suggested_domain == "criminal_law"

    def test_colloquial_haa_routes_to_legal(self):
        decision = route_intent("ما عقوبه السرقه")
        assert decision.intent == IntentType.LEGAL_RETRIEVAL
        assert decision.is_legal_question is True
        assert decision.suggested_domain == "criminal_law"

    def test_both_variants_same_intent(self):
        formal = route_intent("ما عقوبة السرقة")
        colloquial = route_intent("ما عقوبه السرقه")
        assert formal.intent == colloquial.intent

    def test_no_over_normalization_of_natural_haa(self):
        from app.answering.intent_router import _normalize_for_routing
        assert _normalize_for_routing("فيه مشكلة قانونية") == "فيه مشكله قانونيه"
        assert _normalize_for_routing("عليه حكم") == "عليه حكم"
        assert _normalize_for_routing("ايه عقوبة السرقة") == "ايه عقوبه السرقه"
        assert _normalize_for_routing("الله") == "الله"


# ---------------------------------------------------------------------------
# Router unit tests — colloquial / scenario queries
# ---------------------------------------------------------------------------

class TestColloquialRouting:

    def test_colloquial_question_pattern(self):
        decision = route_intent("ايه عقوبة السرقة")
        assert decision.intent == IntentType.LEGAL_RETRIEVAL
        assert decision.is_legal_question is True

    def test_scenario_theft_query(self):
        """A scenario query with the verb سرق should route to retrieval."""
        decision = route_intent("واحد سرق مني موبايل")
        # Should either be LEGAL_RETRIEVAL or AMBIGUOUS,
        # but the service will proceed to retrieval in both cases.
        assert decision.intent in {IntentType.LEGAL_RETRIEVAL, IntentType.AMBIGUOUS}

    def test_colloquial_want_to_know(self):
        decision = route_intent("عايز اعرف عقوبة الضرب")
        assert decision.intent == IntentType.LEGAL_RETRIEVAL
        assert decision.is_legal_question is True


# ---------------------------------------------------------------------------
# Router unit tests — existing behavior preserved
# ---------------------------------------------------------------------------

class TestExistingBehaviorPreserved:

    def test_identity_still_identity(self):
        decision = route_intent("اسمك ايه")
        assert decision.intent == IntentType.IDENTITY

    def test_conversation_still_conversation(self):
        decision = route_intent("السلام عليكم")
        assert decision.intent == IntentType.CONVERSATION

    def test_non_legal_still_non_legal(self):
        decision = route_intent("عايز وصفة اكل")
        assert decision.intent == IntentType.NON_LEGAL

    def test_personal_status_still_external(self):
        decision = route_intent("ما شروط الخلع في مصر")
        assert decision.intent == IntentType.EXTERNAL_ASSISTED

    def test_clear_legal_still_routes_correctly(self):
        decision = route_intent("ما ضمانات الحرية الشخصية في الدستور المصري")
        assert decision.intent == IntentType.LEGAL_RETRIEVAL
        assert decision.suggested_domain == "constitutional_law"


# ---------------------------------------------------------------------------
# Service-level test — ambiguous queries must NOT short-circuit
# ---------------------------------------------------------------------------

class TestAmbiguousGoesToRetrieval:
    """Even if the router says AMBIGUOUS, the service must call retrieval."""

    def test_service_does_not_return_clarification_message(self):
        """The old blocking message should never appear in service output."""
        from app.answering.service import LegalAnswerService

        mock_retriever = MagicMock()
        mock_retriever.search.return_value = {
            "query": "test",
            "normalized_query": "test",
            "rewritten_query": "test",
            "query_analysis": {},
            "results": [],
        }
        mock_llm = MagicMock()
        mock_llm.provider_name = "test"
        mock_llm.model = "test"
        mock_llm.web_search_enabled = False
        mock_llm.chat_completion.side_effect = Exception("no key")

        service = LegalAnswerService(
            retriever=mock_retriever,
            llm_client=mock_llm,
        )

        BLOCKED_MESSAGE = "من فضلك وضّح سؤالك القانوني أو اذكر المجال القانوني المطلوب."

        for query in [
            "ما عقوبه السرقه",
            "واحد سرق مني موبايل",
            "ايه عقوبة السرقة",
        ]:
            response = service.answer(query)
            assert response.final_answer != BLOCKED_MESSAGE, (
                f"Query '{query}' returned the old clarification block"
            )
            # Retriever MUST have been called
            mock_retriever.search.assert_called()

        # Reset and test one more
        mock_retriever.reset_mock()
        response = service.answer("ما عقوبه السرقه")
        # Retriever is called at least once (may be called twice with broad retry)
        assert mock_retriever.search.call_count >= 1
        assert response.router is not None  # Router metadata preserved


# ---------------------------------------------------------------------------
# Generic query classification helpers — unit tests
# ---------------------------------------------------------------------------

class TestLowInformationQuery:
    """_is_low_information_query catches vague/empty/generic inputs."""

    def test_empty_is_low_info(self):
        from app.answering.source_sufficiency import _is_low_information_query
        assert _is_low_information_query("") is True

    def test_punctuation_only_is_low_info(self):
        from app.answering.source_sufficiency import _is_low_information_query
        from app.preprocessing import normalize_legal_arabic
        assert _is_low_information_query(normalize_legal_arabic("؟؟؟")) is True

    def test_two_word_is_low_info(self):
        from app.answering.source_sufficiency import _is_low_information_query
        from app.preprocessing import normalize_legal_arabic
        assert _is_low_information_query(normalize_legal_arabic("اعمل ايه")) is True

    def test_generic_help_request_is_low_info(self):
        from app.answering.source_sufficiency import _is_low_information_query
        from app.preprocessing import normalize_legal_arabic
        assert _is_low_information_query(normalize_legal_arabic("محتاج مساعدة قانونية")) is True
        assert _is_low_information_query(normalize_legal_arabic("ممكن اعرف حقي")) is True
        assert _is_low_information_query(normalize_legal_arabic("عندي قضية اعمل ايه")) is True

    def test_generic_problem_is_low_info(self):
        from app.answering.source_sufficiency import _is_low_information_query
        from app.preprocessing import normalize_legal_arabic
        assert _is_low_information_query(normalize_legal_arabic("فيه مشكلة قانونية")) is True

    def test_concrete_scenario_is_NOT_low_info(self):
        from app.answering.source_sufficiency import _is_low_information_query
        from app.preprocessing import normalize_legal_arabic
        assert _is_low_information_query(normalize_legal_arabic("اعمل ايه لو حد سرق مني الموبايل")) is False
        assert _is_low_information_query(normalize_legal_arabic("ما هو القانون المصري")) is False


class TestConcreteScenario:
    """_is_concrete_legal_scenario detects real-world legal situations."""

    def test_theft_scenario(self):
        from app.answering.source_sufficiency import _is_concrete_legal_scenario
        from app.preprocessing import normalize_legal_arabic
        assert _is_concrete_legal_scenario(normalize_legal_arabic("حد سرق مني الموبايل"), {}) is True

    def test_fraud_scenario(self):
        from app.answering.source_sufficiency import _is_concrete_legal_scenario
        from app.preprocessing import normalize_legal_arabic
        assert _is_concrete_legal_scenario(normalize_legal_arabic("واحد نصب عليا"), {}) is True

    def test_salary_scenario(self):
        from app.answering.source_sufficiency import _is_concrete_legal_scenario
        from app.preprocessing import normalize_legal_arabic
        assert _is_concrete_legal_scenario(normalize_legal_arabic("صاحب الشغل مش مديني مرتبي"), {}) is True

    def test_greeting_is_not_scenario(self):
        from app.answering.source_sufficiency import _is_concrete_legal_scenario
        from app.preprocessing import normalize_legal_arabic
        assert _is_concrete_legal_scenario(normalize_legal_arabic("هاي"), {}) is False

    def test_vague_is_not_scenario(self):
        from app.answering.source_sufficiency import _is_concrete_legal_scenario
        from app.preprocessing import normalize_legal_arabic
        assert _is_concrete_legal_scenario(normalize_legal_arabic("محتاج مساعدة قانونية"), {}) is False


class TestConceptualLegalQuestion:
    """_is_conceptual_legal_question detects definitional / explanatory legal questions."""

    def test_what_is_egyptian_law(self):
        from app.answering.source_sufficiency import _is_conceptual_legal_question
        from app.preprocessing import normalize_legal_arabic
        assert _is_conceptual_legal_question(normalize_legal_arabic("ما هو القانون المصري")) is True

    def test_what_is_civil_law(self):
        from app.answering.source_sufficiency import _is_conceptual_legal_question
        from app.preprocessing import normalize_legal_arabic
        assert _is_conceptual_legal_question(normalize_legal_arabic("ما معنى القانون المدني")) is True

    def test_difference_question(self):
        from app.answering.source_sufficiency import _is_conceptual_legal_question
        from app.preprocessing import normalize_legal_arabic
        assert _is_conceptual_legal_question(normalize_legal_arabic("ما الفرق بين الجناية والجنحة")) is True

    def test_explain_penal_code(self):
        from app.answering.source_sufficiency import _is_conceptual_legal_question
        from app.preprocessing import normalize_legal_arabic
        assert _is_conceptual_legal_question(normalize_legal_arabic("اشرح قانون العقوبات")) is True

    def test_what_is_punishment_meaning(self):
        from app.answering.source_sufficiency import _is_conceptual_legal_question
        from app.preprocessing import normalize_legal_arabic
        assert _is_conceptual_legal_question(normalize_legal_arabic("ما المقصود بالعقوبة")) is True

    def test_vague_legal_help_is_NOT_conceptual(self):
        from app.answering.source_sufficiency import _is_conceptual_legal_question
        from app.preprocessing import normalize_legal_arabic
        assert _is_conceptual_legal_question(normalize_legal_arabic("محتاج مساعدة قانونية")) is False
        assert _is_conceptual_legal_question(normalize_legal_arabic("فيه مشكلة قانونية")) is False
        assert _is_conceptual_legal_question(normalize_legal_arabic("ممكن اعرف حقي")) is False


class TestMeaningfulLegalSignal:
    """_has_meaningful_legal_signal is the single gate for grounding/fallback."""

    def _check(self, query, has_legal_intent=True):
        from app.answering.source_sufficiency import _has_meaningful_legal_signal
        return _has_meaningful_legal_signal(
            query=query, query_analysis={},
            has_legal_intent=has_legal_intent,
            explicit_legal_source_signal=False,
        )

    def test_concrete_scenario_has_signal(self):
        assert self._check("اعمل ايه لو حد سرق مني الموبايل") is True

    def test_conceptual_question_has_signal(self):
        assert self._check("ما هو القانون المصري") is True

    def test_greeting_has_no_signal(self):
        assert self._check("هاي", has_legal_intent=False) is False

    def test_vague_has_no_signal_despite_intent(self):
        assert self._check("محتاج مساعدة قانونية") is False
        assert self._check("عندي قضية اعمل ايه") is False
        assert self._check("فيه مشكلة قانونية") is False

    def test_short_query_has_no_signal(self):
        assert self._check("اعمل ايه") is False

    def test_question_marks_has_no_signal(self):
        assert self._check("؟؟؟", has_legal_intent=False) is False


# ---------------------------------------------------------------------------
# Service-level — external_assisted fallback for meaningful queries
# ---------------------------------------------------------------------------

def _make_empty_retriever_service():
    """Build a LegalAnswerService with a mock retriever returning no results
    and a mock LLM that raises (to isolate mode-decision testing)."""
    mock_retriever = MagicMock()

    def _empty_search(query, *, top_k=None, filters=None):
        return {
            "query": query,
            "normalized_query": query,
            "rewritten_query": query,
            "query_analysis": {},
            "results": [],
        }
    mock_retriever.search.side_effect = _empty_search

    mock_llm = MagicMock()
    mock_llm.provider_name = "test"
    mock_llm.model = "test"
    mock_llm.web_search_enabled = False
    mock_llm.chat_completion.side_effect = Exception("no key")

    from app.answering.service import LegalAnswerService
    service = LegalAnswerService(
        retriever=mock_retriever,
        llm_client=mock_llm,
    )
    return service, mock_retriever


class TestExternalAssistedFallback:
    """Meaningful legal questions with no internal sources → external_assisted."""

    @pytest.mark.parametrize("query", [
        # Concrete scenarios
        "اعمل ايه لو حد سرق مني الموبايل",
        "واحد نصب عليا اعمل ايه",
        "صاحب الشغل مش مديني مرتبي",
        "حد هددني اعمل ايه",
        "مضيت إيصال أمانة ومش عارف أعمل ايه",
        # Conceptual legal questions
        "ما هو القانون المصري",
        "ما معنى القانون المدني",
        "ما الفرق بين الجناية والجنحة",
        "ما المقصود بالعقوبة",
        "اشرح قانون العقوبات",
    ])
    def test_meaningful_legal_gets_external_assisted(self, query):
        service, mock_retriever = _make_empty_retriever_service()
        response = service.answer(query)
        assert response.answer_mode == "external_assisted", (
            f"Query '{query}' got '{response.answer_mode}' instead of 'external_assisted'"
        )
        assert response.internal_grounding_sufficient is False
        assert response.is_supported_by_internal_sources is False
        mock_retriever.search.assert_called()


class TestInsufficientStillInsufficient:
    """Vague / meaningless queries must remain insufficient."""

    @pytest.mark.parametrize("query", [
        "ما هي؟",
        "؟؟؟",
        "فيه مشكلة قانونية",
        "عندي قضية اعمل ايه",
        "محتاج مساعدة قانونية",
        "ممكن اعرف حقي",
        "اعمل ايه",
        "ممكن مساعدة",
    ])
    def test_vague_query_stays_insufficient(self, query):
        service, _ = _make_empty_retriever_service()
        response = service.answer(query)
        assert response.answer_mode == "insufficient", (
            f"Query '{query}' got '{response.answer_mode}' instead of 'insufficient'"
        )


# ---------------------------------------------------------------------------
# Router — greeting classification
# ---------------------------------------------------------------------------

class TestGreetingRouting:
    """Arabic/Egyptian colloquial greetings must be CONVERSATION."""

    @pytest.mark.parametrize("query", [
        "هاي",
        "هاي!",
        "هلو",
        "السلام عليكم",
        "اهلا",
        "أهلا",
        "سلام",
        "مرحبا",
        "تمام",
        "شكرا",
    ])
    def test_greeting_routes_to_conversation(self, query):
        decision = route_intent(query)
        assert decision.intent == IntentType.CONVERSATION, (
            f"Query '{query}' got '{decision.intent}' instead of CONVERSATION"
        )
        assert decision.is_legal_question is False


# ---------------------------------------------------------------------------
# Service — greetings must NOT call retriever or LLM
# ---------------------------------------------------------------------------

class TestGreetingServiceLevel:
    """Greetings must short-circuit: no retriever, no LLM, no sources."""

    @pytest.mark.parametrize("query", [
        "هاي",
        "هلو",
        "السلام عليكم",
    ])
    def test_greeting_does_not_call_retriever(self, query):
        service, mock_retriever = _make_empty_retriever_service()
        response = service.answer(query)
        assert response.answer_mode == "conversation", (
            f"Query '{query}' got '{response.answer_mode}' instead of 'conversation'"
        )
        assert response.llm.called is False
        assert response.internal_sources == []
        assert response.is_legal_question is False
        assert response.is_supported_by_internal_sources is False
        mock_retriever.search.assert_not_called()


# ---------------------------------------------------------------------------
# Safety guard — random Qdrant results must NOT ground non-legal queries
# ---------------------------------------------------------------------------

def _make_fake_legal_sources_retriever():
    """Build a service whose mock retriever returns high-score legal-looking
    results, simulating accidental Qdrant matches for non-legal queries."""
    mock_retriever = MagicMock()

    def _fake_search(query, *, top_k=None, filters=None):
        return {
            "query": query,
            "normalized_query": query,
            "rewritten_query": query,
            "query_analysis": {},
            "results": [
                {
                    "id": "fake-1",
                    "score": 0.92,
                    "law_name": "قانون العقوبات",
                    "law_number": "58",
                    "law_year": "1937",
                    "article_number": "318",
                    "legal_domain": "criminal_law",
                    "title": "سرقة",
                    "content": "يعاقب بالحبس كل من اختلس منقولاً مملوكاً لغيره.",
                    "summary": "عقوبة السرقة البسيطة",
                    "retrieval_text": "يعاقب بالحبس كل من اختلس منقولاً مملوكاً لغيره.",
                },
                {
                    "id": "fake-2",
                    "score": 0.88,
                    "law_name": "قانون العقوبات",
                    "law_number": "58",
                    "law_year": "1937",
                    "article_number": "319",
                    "legal_domain": "criminal_law",
                    "title": "سرقة من حقل",
                    "content": "يعاقب بالحبس مدة لا تتجاوز ستة أشهر.",
                    "summary": "سرقة المحاصيل",
                    "retrieval_text": "يعاقب بالحبس مدة لا تتجاوز ستة أشهر.",
                },
            ],
        }
    mock_retriever.search.side_effect = _fake_search

    mock_llm = MagicMock()
    mock_llm.provider_name = "test"
    mock_llm.model = "test"
    mock_llm.web_search_enabled = False
    mock_llm.chat_completion.side_effect = Exception("no key")

    from app.answering.service import LegalAnswerService
    service = LegalAnswerService(
        retriever=mock_retriever,
        llm_client=mock_llm,
    )
    return service


class TestSafetyGuardNoFalseGrounding:
    """Non-legal queries must NEVER become grounded/assisted even if Qdrant
    returns high-score legal results."""

    @pytest.mark.parametrize("query", [
        "هاي",
        "تمام",
        "شكرا",
    ])
    def test_greeting_not_grounded_despite_sources(self, query):
        service = _make_fake_legal_sources_retriever()
        response = service.answer(query)
        assert response.answer_mode == "conversation", (
            f"Query '{query}' got '{response.answer_mode}' — should be conversation"
        )

    @pytest.mark.parametrize("query", [
        "؟؟؟",
        "محتاج مساعدة قانونية",
    ])
    def test_vague_not_grounded_despite_sources(self, query):
        service = _make_fake_legal_sources_retriever()
        response = service.answer(query)
        assert response.answer_mode not in ("grounded", "assisted"), (
            f"Query '{query}' got '{response.answer_mode}' — must not be grounded/assisted"
        )

