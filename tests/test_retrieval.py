from app.models import RetrievalFilters
from app.retrieval import LegalRetriever


def test_retrieval_expands_chunks_to_parent_article(built_index, fake_embedding_service):
    retriever = LegalRetriever(
        embedding_service=fake_embedding_service,
        client=built_index["client"],
        config=built_index["config"],
    )
    result = retriever.search("ما هي حقوق العامل في عقد العمل؟", filters=RetrievalFilters(legal_domain="labor_law"))

    assert result["results"]
    top = result["results"][0]
    assert top["record_kind"] == "article"
    assert top["id"] == "labor_article_1"
    assert top["supporting_chunks"]


def test_retrieval_can_filter_repealed_records(built_index, fake_embedding_service):
    retriever = LegalRetriever(
        embedding_service=fake_embedding_service,
        client=built_index["client"],
        config=built_index["config"],
    )
    result = retriever.search(
        "ما هي العقوبة الملغاة؟",
        filters=RetrievalFilters(legal_domain="criminal_law", exclude_repealed=True),
    )

    assert result["results"] == []


def test_retrieval_output_contains_legal_metadata(built_index, fake_embedding_service):
    retriever = LegalRetriever(
        embedding_service=fake_embedding_service,
        client=built_index["client"],
        config=built_index["config"],
    )
    result = retriever.search("كيف ينعقد العقد في القانون المدني؟", filters=RetrievalFilters(legal_domain="civil_law"))

    top = result["results"][0]
    assert top["law_number"] == "131"
    assert top["law_year"] == "1948"
    assert top["legal_domain"] == "civil_law"
