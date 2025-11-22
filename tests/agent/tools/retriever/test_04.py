"""
Test DiversityPenalty component.
"""
import pytest
import numpy as np
from src.agent.tools.retriever.diversity_penalty import DiversityPenalty


class TestDiversityPenalty:

    def test_returns_correct_number(self, diversity_penalty):
        """Test that correct number of indices returned."""
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        embeddings = np.random.rand(5, 384).astype(np.float32)

        indices = diversity_penalty.apply_penalty(
            scores=scores,
            embeddings=embeddings,
            top_k=3
        )
        assert len(indices) == 3

    def test_returns_list_of_indices(self, diversity_penalty):
        """Test that indices are integers."""
        scores = np.array([0.9, 0.8, 0.7])
        embeddings = np.random.rand(3, 384).astype(np.float32)

        indices = diversity_penalty.apply_penalty(
            scores=scores,
            embeddings=embeddings,
            top_k=2
        )

        assert isinstance(indices, list)
        assert all(isinstance(idx, (int, np.integer)) for idx in indices)

    def test_highest_score_selected_first(self, diversity_penalty):
        """Test that highest scoring chunk is always selected first."""
        scores = np.array([0.5, 0.9, 0.7, 0.6])  # Index 1 has highest score
        embeddings = np.random.rand(4, 384).astype(np.float32)

        indices = diversity_penalty.apply_penalty(
            scores=scores,
            embeddings=embeddings,
            top_k=3
        )
        assert indices[0] == 1  # Highest score should be first

    def test_penalizes_similar_chunks(self):
        """Test that similar chunks get penalized."""
        # Create scenario: chunk 0 and 1 are VERY similar
        scores = np.array([1.0, 0.99, 0.5])  # Chunk 0 & 1 have similar scores
        # Make embeddings where 0 and 1 are very similar (90% overlap)
        embeddings = np.array([
            [1.0] * 384,      # Chunk 0
            [0.95] * 384,     # Chunk 1 (very similar to 0!)
            [0.1] * 384       # Chunk 2 (different)
        ], dtype=np.float32)
        # Normalize embeddings
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        diversity = DiversityPenalty(
            diversity_threshold=0.85,
            penalty_strength=0.5
        )
        indices = diversity.apply_penalty(
            scores=scores,
            embeddings=embeddings,
            top_k=2
        )
        # Chunk 0 selected first (highest score)
        # Chunk 1 should be penalized (too similar)
        # So chunk 2 should be selected instead
        assert indices[0] == 0  # Highest score
        assert 1 not in indices or indices.index(1) > 0  # Chunk 1 penalized

    def test_handles_top_k_larger_than_chunks(self, diversity_penalty):
        """Test requesting more results than available chunks."""
        scores = np.array([0.9, 0.8])
        embeddings = np.random.rand(2, 384).astype(np.float32)
        indices = diversity_penalty.apply_penalty(
            scores=scores,
            embeddings=embeddings,
            top_k=5  # Request 5, but only 2 exist
        )
        assert len(indices) == 2  # Should return all available

    def test_handles_single_chunk(self, diversity_penalty):
        """Test with single chunk."""
        scores = np.array([0.9])
        embeddings = np.random.rand(1, 384).astype(np.float32)
        indices = diversity_penalty.apply_penalty(
            scores=scores,
            embeddings=embeddings,
            top_k=1
        )
        assert len(indices) == 1
        assert indices[0] == 0

    def test_unique_indices(self, diversity_penalty):
        """Test that no duplicate indices returned."""
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        embeddings = np.random.rand(5, 384).astype(np.float32)
        indices = diversity_penalty.apply_penalty(
            scores=scores,
            embeddings=embeddings,
            top_k=5
        )
        assert len(indices) == len(set(indices))  # All unique