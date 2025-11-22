"""
Test HybridScores dataclass validation.
"""
import pytest
import numpy as np
from src.agent.tools.retriever.base import HybridScores


class TestHybridScores:

    def test_create_valid_scores(self):
        """Test creating HybridScores with valid data."""
        bm25 = np.array([0.8, 0.6, 0.4])
        semantic = np.array([0.9, 0.7, 0.5])
        fused = np.array([0.85, 0.65, 0.45])
        final = np.array([0.92, 0.71, 0.50])

        scores = HybridScores(
            bm25_scores=bm25,
            semantic_scores=semantic,
            fused_scores=fused,
            final_scores=final
        )
        assert len(scores.bm25_scores) == 3
        assert len(scores.semantic_scores) == 3
        assert len(scores.fused_scores) == 3
        assert len(scores.final_scores) == 3

    def test_scores_numpy_arrays(self):
        """Test that all scores are numpy arrays."""
        bm25 = np.array([0.8, 0.6])
        semantic = np.array([0.9, 0.7])
        fused = np.array([0.85, 0.65])
        final = np.array([0.92, 0.71])

        scores = HybridScores(
            bm25_scores=bm25,
            semantic_scores=semantic,
            fused_scores=fused,
            final_scores=final
        )
        assert isinstance(scores.bm25_scores, np.ndarray)
        assert isinstance(scores.semantic_scores, np.ndarray)
        assert isinstance(scores.fused_scores, np.ndarray)
        assert isinstance(scores.final_scores, np.ndarray)

    def test_mismatched_lengths(self):
        """Test that mismatched array lengths raise ValueError."""
        bm25 = np.array([0.8, 0.6, 0.4])  # 3 elements
        semantic = np.array([0.9, 0.7])    # 2 elements (mismatch!)
        fused = np.array([0.85, 0.65, 0.45])
        final = np.array([0.92, 0.71, 0.50])

        with pytest.raises(ValueError, match="same length"):
            HybridScores(
                bm25_scores=bm25,
                semantic_scores=semantic,
                fused_scores=fused,
                final_scores=final
            )

    def test_different_lengths_array(self):
        """Test that all different lengths raise ValueError."""
        bm25 = np.array([0.8, 0.6])           # 2 elements
        semantic = np.array([0.9, 0.7, 0.5])  # 3 elements
        fused = np.array([0.85, 0.65, 0.45, 0.25])  # 4 elements
        final = np.array([0.92])              # 1 element

        with pytest.raises(ValueError):
            HybridScores(
                bm25_scores=bm25,
                semantic_scores=semantic,
                fused_scores=fused,
                final_scores=final
            )

    def test_access_individual_scores(self):
        """Test accessing individual scores by index."""
        bm25 = np.array([0.8, 0.6, 0.4])
        semantic = np.array([0.9, 0.7, 0.5])
        fused = np.array([0.85, 0.65, 0.45])
        final = np.array([0.92, 0.71, 0.50])

        scores = HybridScores(
            bm25_scores=bm25,
            semantic_scores=semantic,
            fused_scores=fused,
            final_scores=final
        )
        # Access scores for chunk at index 0
        assert scores.bm25_scores[0] == 0.8
        assert scores.semantic_scores[0] == 0.9
        assert scores.fused_scores[0] == 0.85
        assert scores.final_scores[0] == 0.92