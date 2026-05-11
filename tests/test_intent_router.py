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
# Source sufficiency — understandable legal question heuristic
# ---------------------------------------------------------------------------

class TestUnderstandableLegalQuestionHeuristic:
    """Test _is_understandable_legal_question directly."""

    def test_scenario_with_intent_is_understandable(self):
        from app.answering.source_sufficiency import _is_understandable_legal_question
        assert _is_understandable_legal_question("اعمل ايه لو حد سرق مني الموبايل", has_legal_intent=True) is True

    def test_short_vague_query_is_not_understandable(self):
        from app.answering.source_sufficiency import _is_understandable_legal_question
        assert _is_understandable_legal_question("ما هي", has_legal_intent=True) is False

    def test_two_word_query_is_not_understandable(self):
        from app.answering.source_sufficiency import _is_understandable_legal_question
        assert _is_understandable_legal_question("اعمل ايه", has_legal_intent=True) is False

    def test_question_marks_only_is_not_understandable(self):
        from app.answering.source_sufficiency import _is_understandable_legal_question
        assert _is_understandable_legal_question("؟؟؟", has_legal_intent=False) is False

    def test_scenario_without_intent_but_with_cues(self):
        from app.answering.source_sufficiency import _is_understandable_legal_question
        assert _is_understandable_legal_question("واحد نصب عليا اعمل ايه", has_legal_intent=False) is True

    def test_scenario_theft_is_understandable(self):
        from app.answering.source_sufficiency import _is_understandable_legal_question
        assert _is_understandable_legal_question("حد سرق مني موبايل اعمل ايه", has_legal_intent=True) is True

    def test_vague_legal_without_scenario_cue_is_not_understandable(self):
        """Legal intent alone is NOT enough — must have a concrete scenario cue."""
        from app.answering.source_sufficiency import _is_understandable_legal_question
        assert _is_understandable_legal_question("فيه مشكلة قانونية", has_legal_intent=True) is False
        assert _is_understandable_legal_question("عندي قضية اعمل ايه", has_legal_intent=True) is False
        assert _is_understandable_legal_question("محتاج مساعدة قانونية", has_legal_intent=True) is False
        assert _is_understandable_legal_question("ممكن اعرف حقي", has_legal_intent=True) is False


# ---------------------------------------------------------------------------
# Service-level — external_assisted fallback for understandable queries
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
    """Understandable legal questions with no internal sources → external_assisted, not insufficient."""

    @pytest.mark.parametrize("query", [
        "اعمل ايه لو حد سرق مني الموبايل",
        "واحد نصب عليا اعمل ايه",
        "صاحب الشغل مش مديني مرتبي",
        "حد هددني اعمل ايه",
        "مضيت إيصال أمانة ومش عارف أعمل ايه",
    ])
    def test_understandable_legal_gets_external_assisted(self, query):
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
    ])
    def test_vague_query_stays_insufficient(self, query):
        service, _ = _make_empty_retriever_service()
        response = service.answer(query)
        assert response.answer_mode == "insufficient", (
            f"Query '{query}' got '{response.answer_mode}' instead of 'insufficient'"
        )

