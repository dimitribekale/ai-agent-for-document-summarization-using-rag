"""
Retrieval pipeline.
"""
import numpy as np
from typing import List, Optional
from src.agent.tools.base import Tool
from src.agent.tools.retriever.base import (
    RetrieverInput,
    RetrieverOutput,
    RetrieverResult,
    ChunkMetadata,
    HybridScores,
    )
from src.agent.tools.retriever.hybrid_retrieval import HybridRetriever
from src.agent.tools.retriever.diversity_penalty import DiversityPenalty
from src.agent.errors import RetrievalError
from src.logging_config import get_logger

logger = get_logger(__name__)

class Retriever(Tool):
    """Retrieves diverse and relevant chunks for a query."""
    def __init__(self,
                 hybrid_retrieval: Optional[HybridRetriever] = None,
                 diversity_penalty: Optional[DiversityPenalty] = None
                 ):
        super().__init__()
        self.hybrid_retrieval = hybrid_retrieval or HybridRetriever()
        self.diversity_penalty = diversity_penalty or DiversityPenalty()
        logger.info("Retriever initialized")
    
    def name(self) -> str:
        return "Retriever"

    def execute(self, input_data: RetrieverInput) -> RetrieverOutput:
        """
        Retriver pipeline: relevance -> diversity -> results.
        Returns:
            RetrieverOutput with ranked, diverse chunks.
        """
        try:
            self._validate_input(input_data)
            logger.info(f"Retrieving for query: '{input_data.query}'")
            embeddings_array = np.array(input_data.embeddings)
            hybrid_scores = self.hybrid_retrieval.retrieve(
                query=input_data.query,
                chunks=input_data.chunks,
                embeddings=embeddings_array,
                bm25_weight=input_data.bm25_weight,
                semantic_weight=input_data.semantic_weight,
                rerank_top_n=input_data.top_k * 2 # Rerank more candidates
            )
            diverse_indices = self.diversity_penalty.apply_penalty(
                scores=hybrid_scores.final_scores,
                embeddings=embeddings_array,
                top_k=input_data.top_k,
            )
            results = self._build_results(
                indices=diverse_indices,
                chunks=input_data.chunks,
                hybrid_scores=hybrid_scores
            )
            logger.info(f"Retrieved {len(results)} diverse results.")
            return RetrieverOutput(
                results=results,
                results_count=len(results),
                query_expanded=False,
                expanded_query_terms=[],
                reranking_applied=True,
                processing_stages={}
            )
        
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            raise RetrievalError(f"Retrieval pipeline failed: {e}") from e
        
    def _validate_input(self, input_data: RetrieverInput) -> None:
        if len(input_data.chunks) == 0:
            raise RetrievalError("No chunks provided for retrieval")
        
        if len(input_data.chunks) != len(input_data.embeddings):
            raise RetrievalError(
                f"Chunks count ({len(input_data.chunks)}) must match "
                f"embeddings count ({len(input_data.embeddings)})"
            )
        if input_data.top_k <= 0:
            raise RetrievalError(f"top_k must be positive, got {input_data.top_k}")
        logger.debug("Input validation passed")

    def _build_results(self,
                       indices: List[int],
                       chunks: List[str],
                       hybrid_scores: HybridScores ) -> List[RetrieverResult]:
        """Build RetrieverResult objects"""
        results = []
        for rank, idx in enumerate(indices):
          # Create metadata (simple version from base.py)
            metadata = ChunkMetadata(
                chunk_index=idx,
                source_position="middle",
                chunk_length=len(chunks[idx]),
                is_header=False
            )
            # Create result with ALL fields from base.py
            result = RetrieverResult(
                chunk=chunks[idx],
                chunk_index=idx,
                bm25_score=float(hybrid_scores.bm25_scores[idx]),
                semantic_score=float(hybrid_scores.semantic_scores[idx]),
                bm25_rank=rank + 1,
                semantic_rank=rank + 1,
                rrf_score=float(hybrid_scores.fused_scores[idx]),
                rerank_score=float(hybrid_scores.final_scores[idx]),
                final_score=float(hybrid_scores.final_scores[idx]),
                diversity_penality=0.0,  # Note: typo in base.py!
                metadata=metadata
            )
            results.append(result)

        logger.debug(f"Built {len(results)} result objects")
        return results