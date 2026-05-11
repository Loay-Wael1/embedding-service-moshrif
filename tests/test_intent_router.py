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
