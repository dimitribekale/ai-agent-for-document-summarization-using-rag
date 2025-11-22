"""
Test HybridRetrieval component.
"""
import pytest
import numpy as np
from src.agent.tools.retriever.hybrid_retrieval import HybridRetriever
from src.agent.tools.retriever.base import HybridScores


class TestHybridRetrieval:

    def test_returns_hybrid_scores(self, hybrid_retrieval, sample_query, sample_chunks, sample_embeddings):
        """Test that retrieve returns HybridScores object."""
        result = hybrid_retrieval.retrieve(
            query=sample_query,
            chunks=sample_chunks,
            embeddings=sample_embeddings
        )
        assert isinstance(result, HybridScores)

    def test_all_scores_same_length(self, hybrid_retrieval, sample_query, sample_chunks, sample_embeddings):
        """Test that all score arrays match chunks length."""
        result = hybrid_retrieval.retrieve(
            query=sample_query,
            chunks=sample_chunks,
            embeddings=sample_embeddings
        )
        expected_length = len(sample_chunks)
        assert len(result.bm25_scores) == expected_length
        assert len(result.semantic_scores) == expected_length
        assert len(result.fused_scores) == expected_length
        assert len(result.final_scores) == expected_length

    def test_scores_are_normalized(self, hybrid_retrieval, sample_query, sample_chunks, sample_embeddings):
        """Test that scores are in reasonable range [0, 1]."""
        result = hybrid_retrieval.retrieve(
            query=sample_query,
            chunks=sample_chunks,
            embeddings=sample_embeddings
        )
        # All scores should be between 0 and 1
        assert np.all(result.bm25_scores >= 0)
        assert np.all(result.bm25_scores <= 1)
        assert np.all(result.semantic_scores >= 0)
        assert np.all(result.semantic_scores <= 1)
        assert np.all(result.fused_scores >= 0)
        assert np.all(result.fused_scores <= 1)
        assert np.all(result.final_scores >= 0)
        assert np.all(result.final_scores <= 1)

    def test_relevant_chunk_scores_higher(self, hybrid_retrieval, sample_embeddings):
        """Test that relevant chunks score higher than irrelevant ones."""
        chunks = [
            "Climate change causes global warming.",  # Relevant
            "Pizza is a delicious Italian food.",     # Irrelevant
            "Global warming affects ice caps."        # Relevant
        ]
        query = "climate change effects"
        result = hybrid_retrieval.retrieve(
            query=query,
            chunks=chunks,
            embeddings=sample_embeddings[:3]
        )
        # Relevant chunks (0, 2) should score higher than irrelevant (1)
        assert result.final_scores[0] > result.final_scores[1]
        assert result.final_scores[2] > result.final_scores[1]

    def test_custom_weights(self, hybrid_retrieval, sample_query, sample_chunks, sample_embeddings):
        """Test that custom weights are applied."""
        result1 = hybrid_retrieval.retrieve(
            query=sample_query,
            chunks=sample_chunks,
            embeddings=sample_embeddings,
            bm25_weight=1.0,
            semantic_weight=0.0
        )
        result2 = hybrid_retrieval.retrieve(
            query=sample_query,
            chunks=sample_chunks,
            embeddings=sample_embeddings,
            bm25_weight=0.0,
            semantic_weight=1.0
        )
        # Different weights should produce different results
        assert not np.array_equal(result1.fused_scores, result2.fused_scores)

    def test_handles_single_chunk(self, hybrid_retrieval):
        """Test retrieval with single chunk."""
        chunks = ["Climate change is a global issue."]
        embeddings = np.random.rand(1, 384).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        result = hybrid_retrieval.retrieve(
            query="climate change",
            chunks=chunks,
            embeddings=embeddings
        )
        assert len(result.final_scores) == 1
        assert 0 <= result.final_scores[0] <= 1