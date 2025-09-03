import pytest
from ingest_to_mongodb import chunk_text

def test_chunk_text_basic():
    text = "a" * 2100
    chunks = chunk_text(text, size=2000, overlap=200)
    assert len(chunks) == 2
    assert len(chunks[0]) == 2000
