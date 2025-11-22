"""
Diversity penalty component for reducing redundancy in search results.
"""
import numpy as np
from typing import List


class DiversityPenalty:
    """
    Applies penalties to chunks that are too similar to
    already-selected results.

    Args:
        diversity_threshold: Similarity above this = "too similar" (0-1)
        penalty_strength: How much to reduce score (0-1, higher = stronger penalty)
    """
    def __init__(
            self,
            diversity_threshold: float = 0.85,
            penalty_strength: float = 0.5
    ):
        self.diversity_threshold = diversity_threshold
        self.penalty_strength = penalty_strength

    def apply_penalty(self, scores: np.ndarray, embeddings: np.ndarray, top_k: int = 5) -> List[int]:
        """
        Select top-k diverse results by penalizing similar chunks
        
        Args:
            scores: Similarity scores for each chunk (1D array)
            embeddings: Chunk embedding (2D array: [num_chunks, embedding_dim])
            top_k: How many results to return
            
        Return:
            Indices of top-k diverse chunks
        """
        selected_indices = []
        remaining_indices = list(range(len(scores)))
        adjusted_scores = scores.copy() # Not mutate original

        for _ in range(min(top_k, len(scores))):
            best_idx = remaining_indices[np.argmax(adjusted_scores[remaining_indices])]
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

            if not remaining_indices:
                break
            # Penalize remaining chunks similar to the one that was selected
            selected_embedding = embeddings[best_idx]
            for idx in remaining_indices:
                similarity = self._cosine_similarity(
                    selected_embedding,
                    embeddings[idx]
                )
                # Apply similarity
                if similarity > self.diversity_threshold:
                    adjusted_scores[idx] *= (1 - self.penalty_strength)

        return selected_indices
    
    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        Returns: Float in range [-1, 1], but embeddings typically [0, 1]
        """
        dot_product = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)

        if norm_product == 0:
            return 0.0
        return float(dot_product / norm_product)