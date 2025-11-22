"""
Shared fixtures for retriever tests.
"""
import pytest
import numpy as np
from src.agent.tools.retriever import Retriever, RetrieverInput
from src.agent.tools.retriever.hybrid_retrieval import HybridRetriever
from src.agent.tools.retriever.diversity_penalty import DiversityPenalty


@pytest.fixture
def retriever(config):
    """Creates a retriever instance."""
    return Retriever()


@pytest.fixture
def sample_chunks():
    """Sample text chunks for testing."""
    return [
        "Climate change causes global warming and rising sea levels.",
        "Machine learning algorithms learn patterns from data.",
        "Global warming affects polar ice caps and weather patterns.",
        "Deep learning uses neural networks with multiple layers.",
        "Rising temperatures impact ecosystems and biodiversity."
    ]


@pytest.fixture
def sample_embeddings():
    """Sample embeddings (384-dim) for chunks."""
    np.random.seed(42)
    embeddings = np.random.rand(5, 384).astype(np.float32)
    # Normalize to unit vectors (like real embeddings)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings


@pytest.fixture
def sample_query():
    """Sample query for testing."""
    return "climate change effects"


@pytest.fixture
def hybrid_retrieval():
    """Creates a HybridRetrieval instance."""
    return HybridRetriever()


@pytest.fixture
def diversity_penalty():
    """Creates a DiversityPenalty instance."""
    return DiversityPenalty()