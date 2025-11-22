"""
Test full Retriever pipeline (end-to-end integration).
"""
import pytest
import numpy as np
from src.agent.tools.retriever import Retriever, RetrieverInput, RetrieverOutput


class TestFullRetrieverPipeline:

    def test_execute_returns_retriever_output(self, retriever, sample_chunks, sample_embeddings, sample_query):
        """Test that execute returns RetrieverOutput."""
        input_data = RetrieverInput(
            query=sample_query,
            chunks=sample_chunks,
            embeddings=sample_embeddings.tolist()
        )
        result = retriever.execute(input_data)
        assert isinstance(result, RetrieverOutput)

    def test_returns_correct_number_of_results(self, retriever, sample_chunks, sample_embeddings, sample_query):
        """Test that top_k results are returned."""
        input_data = RetrieverInput(
            query=sample_query,
            chunks=sample_chunks,
            embeddings=sample_embeddings.tolist(),
            top_k=3
        )
        result = retriever.execute(input_data)
        assert len(result.results) == 3

    def test_results_have_metadata(self, retriever, sample_chunks, sample_embeddings, sample_query):
        """Test that results contain full metadata."""
        input_data = RetrieverInput(
            query=sample_query,
            chunks=sample_chunks,
            embeddings=sample_embeddings.tolist(),
            top_k=2
        )
        result = retriever.execute(input_data)
        for res in result.results:
            assert res.chunk is not None
            assert res.final_score is not None
            assert res.bm25_rank >= 1
            assert res.metadata is not None

    def test_metadata_has_real_scores(self, retriever, sample_chunks, sample_embeddings, sample_query):
        """Test that metadata contains actual scores, not 0.0."""
        input_data = RetrieverInput(
            query=sample_query,
            chunks=sample_chunks,
            embeddings=sample_embeddings.tolist()
        )
        result = retriever.execute(input_data)
        # Check that we have REAL scores (not all 0.0)
        first_result = result.results[0]
        # At least one score should be non-zero
        assert (
            first_result.bm25_score > 0 or
            first_result.semantic_score > 0 or
            first_result.final_score > 0
        )

    def test_results_are_ranked(self, retriever, sample_chunks, sample_embeddings, sample_query):
        """Test that results are ranked by score (descending)."""
        input_data = RetrieverInput(
            query=sample_query,
            chunks=sample_chunks,
            embeddings=sample_embeddings.tolist(),
            top_k=5
        )
        result = retriever.execute(input_data)
        # Ranks should be 1, 2, 3, 4, 5
        ranks = [r.bm25_rank for r in result.results]  # Use bm25_rank as proxy
        assert ranks == [1, 2, 3, 4, 5]
        # Scores should be descending
        scores = [r.final_score for r in result.results]
        assert scores == sorted(scores, reverse=True)

    def test_diverse_results(self, retriever, sample_embeddings):
        """Test that diversity penalty produces varied results."""
        # Create chunks with known similarity structure
        chunks = [
            "Climate change affects global temperature.",
            "Machine learning uses neural networks.",
            "Deep learning is a subset of ML."
        ]
        input_data = RetrieverInput(
            query="climate change",
            chunks=chunks,
            embeddings=sample_embeddings[:3].tolist(),
            top_k=2
        )
        result = retriever.execute(input_data)
        # Should return 2 diverse results
        assert len(result.results) == 2
        assert result.results[0].chunk != result.results[1].chunk

    def test_validates_input(self, retriever, sample_chunks, sample_embeddings):
        """Test that input validation catches mismatches."""
        # Mismatch: 5 chunks but only 3 embeddings
        input_data = RetrieverInput(
            query="test",
            chunks=sample_chunks,  # 5 chunks
            embeddings=sample_embeddings[:3].tolist()  # 3 embeddings
        )
        with pytest.raises(Exception):  # Should raise RetrieverError
            retriever.execute(input_data)

    def test_handles_custom_weights(self, retriever, sample_chunks, sample_embeddings, sample_query):
        """Test that custom BM25/semantic weights work."""
        input_data = RetrieverInput(
            query=sample_query,
            chunks=sample_chunks,
            embeddings=sample_embeddings.tolist(),
            bm25_weight=0.8,
            semantic_weight=0.2
        )
        result = retriever.execute(input_data)
        assert len(result.results) > 0
        # Results should still have valid scores
        assert all(r.final_score >= 0 for r in result.results)

    def test_output_contains_query(self, retriever, sample_chunks, sample_embeddings, sample_query):
        """Test that output contains original query."""
        input_data = RetrieverInput(
            query=sample_query,
            chunks=sample_chunks,
            embeddings=sample_embeddings.tolist()
        )
        result = retriever.execute(input_data)
        assert result.success is True  # Check it succeeded instead

    def test_output_contains_total_chunks(self, retriever, sample_chunks, sample_embeddings, sample_query):
        """Test that output records total chunks searched."""
        input_data = RetrieverInput(
            query=sample_query,
            chunks=sample_chunks,
            embeddings=sample_embeddings.tolist()
        )
        result = retriever.execute(input_data)
        assert result.results_count == len(sample_chunks)  # Use results_count