from app.indexing import load_legal_dataset


def test_dataset_loader_creates_law_article_and_chunk_records(sample_dataset_path):
    records = load_legal_dataset(sample_dataset_path, include_law_records=True)
    kinds = {record.record_kind for record in records}

    assert "law" in kinds
    assert "article" in kinds
    assert "article_chunk" in kinds


def test_index_builder_creates_new_legal_collection(built_index):
    summary = built_index["summary"]
    client = built_index["client"]
    collection = client.get_collection(summary.collection_name)

    assert summary.total_records == 8
    assert summary.article_count == 3
    assert summary.chunk_count == 2
    assert summary.law_count == 3
    assert collection.points_count == 8
