"""
Test RetrieverInput validation.
"""
import pytest
import numpy as np
from src.agent.tools.retriever import RetrieverInput


class TestInputValidation:

    def test_valid_input(self, sample_chunks, sample_embeddings):
        """Test valid input creation."""
        input_data = RetrieverInput(
            query="test query",
            chunks=sample_chunks,
            embeddings=sample_embeddings.tolist()
        )
        assert input_data.query == "test query"
        assert len(input_data.chunks) == 5
        assert input_data.top_k == 5  # Default value

    def test_custom_top_k(self, sample_chunks, sample_embeddings):
        """Test custom top_k parameter."""
        input_data = RetrieverInput(
            query="test",
            chunks=sample_chunks,
            embeddings=sample_embeddings.tolist(),
            top_k=3
        )
        assert input_data.top_k == 3

    def test_custom_weights(self, sample_chunks, sample_embeddings):
        """Test custom BM25 and semantic weights."""
        input_data = RetrieverInput(
            query="test",
            chunks=sample_chunks,
            embeddings=sample_embeddings.tolist(),
            bm25_weight=0.3,
            semantic_weight=0.7
        )
        assert input_data.bm25_weight == 0.3
        assert input_data.semantic_weight == 0.7

    def test_require_query(self, sample_chunks, sample_embeddings):
        """Input requires query parameter."""
        with pytest.raises(Exception):
            RetrieverInput(
                chunks=sample_chunks,
                embeddings=sample_embeddings.tolist()
            )

    def test_require_chunks(self, sample_embeddings):
        """Input requires chunks parameter."""
        with pytest.raises(Exception):
            RetrieverInput(
                query="test",
                embeddings=sample_embeddings.tolist()
            )

    def test_require_embeddings(self, sample_chunks):
        """Input requires embeddings parameter."""
        with pytest.raises(Exception):
            RetrieverInput(
                query="test",
                chunks=sample_chunks
            )

    def test_validate_consistency(self, sample_chunks, sample_embeddings):
        """Test that chunks and embeddings counts match."""
        input_data = RetrieverInput(
            query="test",
            chunks=sample_chunks,
            embeddings=sample_embeddings.tolist()
        )
        assert input_data.validate_consistency() is True

    def test_validate_consistency_mismatch(self, sample_chunks, sample_embeddings):
        """Test validation fails when counts mismatch."""
        input_data = RetrieverInput(
            query="test",
            chunks=sample_chunks[:3],  # Only 3 chunks
            embeddings=sample_embeddings.tolist()  # 5 embeddings
        )
        assert input_data.validate_consistency() is False