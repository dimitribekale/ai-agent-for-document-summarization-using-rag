"""
Combined lexical (BM25) and semantic search as Hybrid retrieval.
"""
import numpy as np
from typing import List, Optional
from src.agent.tools.retriever.base import HybridScores
from src.agent.tools.retriever.bm25 import BM25Scorer
from src.agent.tools.retriever.semantic_search import SemanticSearcher
from src.agent.tools.retriever.rrf_fusion import RRFFusion
from src.agent.tools.retriever.query_expander import QueryExpander
from src.agent.tools.retriever.reranker import Reranker
from src.logging_config import get_logger

logger = get_logger(__name__)

class HybridRetriever:
    """
    Combines multiple retrieval methods into a unified relevance score.

    Methods:
        - BM25: Lexical matching (keyword frequency)
        - Semantic: Embedding similarity (meaning)
        - Query expansion: Enriches query with related terms
        - Cross-encoder reranking: Refines top candidates
        - RRF fusion: Mathematically combines rankings
    """
    def __init__(
        self,
        bm25_scorer: Optional[BM25Scorer] = None,
        semantic_searcher: Optional[SemanticSearcher] = None,
        rrf_fusion: Optional[RRFFusion] = None,
        query_expander: Optional[QueryExpander] = None,
        cross_encoder_reranker: Optional[Reranker] = None
      ):
        
        self.bm25_scorer = bm25_scorer or BM25Scorer()
        self.semantic_searcher = semantic_searcher or SemanticSearcher()
        self.rrf_fusion = rrf_fusion or RRFFusion()
        self.query_expander = query_expander or QueryExpander()
        self.cross_encoder_reranker = cross_encoder_reranker or Reranker()

        logger.info("HybridRetrieval initialized")

    def retrieve(
            self,
            query: str,
            chunks: List[str],
            embeddings: np.ndarray,
            bm25_weight: float = 0.5,
            semantic_weight: float = 0.5,
            rerank_top_n: int = 20 ) -> HybridScores:
        
        logger.info(f"Starting hybrid retrieval for query: '{query}'")
        expanded_query = self._expand_query(query, chunks, embeddings)
        bm25_scores = self._compute_bm25_scores(expanded_query, chunks)
        semantic_scores = self._compute_semantic_scores(expanded_query, embeddings)
        fused_scores = self._fuse_scores(
            bm25_scores,
            semantic_scores,
            bm25_weight,
            semantic_weight
        )
        final_scores = self._rerank_top_candidates(
            expanded_query,
            chunks,
            fused_scores,
            rerank_top_n,
        )
        logger.info(f"Hybrid retrieval complete. Max score: {np.max(final_scores):.3f}")
        return HybridScores(
            bm25_scores=bm25_scores,
            semantic_scores=semantic_scores,
            fused_scores=fused_scores,
            final_scores=final_scores,
        )
    
    def _expand_query(self, query: str, chunks: List[str], embeddings: np.ndarray) -> str:
        """Returns original if fails"""
        try:
            expanded = self.query_expander.expand(query, chunks, embeddings)
            logger.debug(f"Query expansion: '{query}' and '{expanded}'")
            return expanded
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}. Using original query.")
            return query
        
    def _compute_bm25_scores(self, query: str, chunks: List[str]) -> np.ndarray:
        scores = self.bm25_scorer.score(query, chunks)
        logger.debug(f"BM25 scores - min: {np.min(scores):.3f}, max: {np.max(scores):.3f}")
        return scores
    
    def _compute_semantic_scores(self, query: str, embeddings: np.ndarray) -> np.ndarray:
        scores = self.semantic_searcher.score(query, embeddings)
        logger.debug(f"Semantic scores - min: {np.min(scores):.3f}, max: {np.max(scores):.3f}")
        return scores
    
    def _fuse_scores(self,
                     bm25_scores: np.ndarray,
                     semantic_scores: np.ndarray,
                     bm25_weight: float,
                     semantic_weight: float
                     ) -> np.ndarray:
        fused = self.rrf_fusion.fuse(
            bm25_scores,
            semantic_scores,
            bm25_weight,
            semantic_weight
        )
        logger.debug(f"Fused scores - min: {fused.min():.3f}, max: {fused.max():.3f}")
        return fused
    
    def _rerank_top_candidates(self,
                               query: str,
                               chunks: List[str],
                               scores: np.ndarray,
                               top_n: int ) -> np.ndarray:
        try:
            top_indices = np.argsort(scores)[::-1][:top_n]
            top_chunks = [chunks[i] for i in top_indices]
            reranked_scores = self.cross_encoder_reranker.rerank(query, top_chunks)

            updated_scores = scores.copy()
            for idx, new_score in zip(top_indices, reranked_scores):
                updated_scores[idx] = new_score

            logger.debug(f"Reranked top {len(top_indices)} candidates.")
            return updated_scores
        except Exception as e:
            logger.warning(f"Reranking failed: {e}. Using fused scores.")
            return scores # Fallback to original scores
        