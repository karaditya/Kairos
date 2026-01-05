"""
DrBERT RAG Module - Clean Parlant-Native Implementation

DrBERT + ChromaDB for semantic search over medical protocols.
Designed as a Parlant tool backend with clean, simple interfaces.
"""

import os
import gc
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from config import (
    DRBERT_MODELS,
    DEFAULT_DRBERT_MODEL,
    CHROMA_PERSIST_DIR,
    ENGLISH_PROTOCOLS_COLLECTION,
    FRENCH_PROTOCOLS_COLLECTION,
    PATIENT_CASES_COLLECTION,
    CONFIG_DIR,
)

logger = logging.getLogger(__name__)

# =============================================================================
# LAZY IMPORTS (Optional Dependencies)
# =============================================================================

TORCH_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False
CHROMADB_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    pass

try:
    from transformers import AutoModel, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    pass

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Protocol:
    """A medical protocol from the vector store."""
    id: str
    title: str
    text: str
    source: str
    category: str
    language: str
    relevance_score: float = 0.0
    keywords: List[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "text": self.text,
            "source": self.source,
            "category": self.category,
            "language": self.language,
            "relevance_score": self.relevance_score,
            "keywords": self.keywords or [],
        }


@dataclass
class SearchResult:
    """Result from a protocol search."""
    protocols: List[Protocol]
    query_used: str
    language: str
    total_found: int


# =============================================================================
# LANGUAGE DETECTION
# =============================================================================

FRENCH_KEYWORDS = [
    "douleur", "fièvre", "tête", "ventre", "essoufflement", "poitrine",
    "thoracique", "mal", "nausée", "vomissement", "fatigue", "vertige",
    "palpitations", "gonflement", "saignement", "respiration", "cœur",
    "abdomen", "gorge", "toux", "frissons", "sueur", "perte"
]

ENGLISH_KEYWORDS = [
    "pain", "fever", "head", "stomach", "breath", "chest", "ache",
    "nausea", "vomiting", "fatigue", "dizziness", "palpitations",
    "swelling", "bleeding", "breathing", "heart", "abdomen", "throat",
    "cough", "chills", "sweating", "loss", "shortness", "pressure"
]


def detect_language(text: str) -> str:
    """
    Detect language of medical text (English or French).

    Args:
        text: Text to analyze

    Returns:
        'en' for English, 'fr' for French
    """
    text_lower = text.lower()
    french_count = sum(1 for kw in FRENCH_KEYWORDS if kw in text_lower)
    english_count = sum(1 for kw in ENGLISH_KEYWORDS if kw in text_lower)
    return "fr" if french_count > english_count else "en"


# =============================================================================
# DRBERT RAG CLASS
# =============================================================================

class DrBERTRAG:
    """
    DrBERT + ChromaDB for medical protocol retrieval.

    Clean implementation designed for Parlant tool integration.

    Features:
    - Dual-language support (English/French)
    - GPU/CPU automatic device mapping
    - Semantic similarity search
    - Protocol ingestion and management
    """

    def __init__(self, init_vector_store: bool = True):
        """
        Initialize DrBERT RAG.

        Args:
            init_vector_store: Initialize ChromaDB immediately (default: True)
        """
        self.model = None
        self.tokenizer = None
        self.model_id: Optional[str] = None
        self.device = None
        self._chroma_client = None
        self._english_collection = None
        self._french_collection = None
        self._case_collection = None

        # Initialize vector store immediately so it's available even before model loads
        if init_vector_store:
            self._init_vector_store()

    @property
    def is_loaded(self) -> bool:
        """Check if DrBERT model is loaded."""
        return self.model is not None

    @property
    def rag_available(self) -> bool:
        """Check if RAG is ready (model + vector store)."""
        return self.is_loaded and self.vector_store_available

    @property
    def vector_store_available(self) -> bool:
        """Check if ChromaDB is available."""
        return CHROMADB_AVAILABLE and self._chroma_client is not None

    def load_model(self, model_id: str = DEFAULT_DRBERT_MODEL) -> bool:
        """
        Load DrBERT model with automatic GPU/CPU device mapping.

        Args:
            model_id: Model ID from DRBERT_MODELS

        Returns:
            True if loaded successfully
        """
        if not TRANSFORMERS_AVAILABLE or not TORCH_AVAILABLE:
            logger.error("transformers or torch not available")
            return False

        if model_id not in DRBERT_MODELS:
            logger.error(f"Unknown model: {model_id}")
            return False

        config = DRBERT_MODELS[model_id]
        logger.info(f"Loading DrBERT model: {config.name}")

        try:
            # Determine device
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info("Using CUDA GPU")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
                logger.info("Using Apple MPS")
            else:
                self.device = torch.device("cpu")
                logger.info("Using CPU")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(config.hf_repo)

            # Load model
            self.model = AutoModel.from_pretrained(
                config.hf_repo,
                torch_dtype=torch.float16 if self.device.type != "cpu" else torch.float32
            )
            self.model = self.model.to(self.device)
            self.model.eval()

            self.model_id = model_id
            logger.info(f"DrBERT loaded successfully on {self.device}")

            # Initialize vector store if not already initialized
            if not self.vector_store_available:
                self._init_vector_store()

            return True

        except Exception as e:
            logger.error(f"Failed to load DrBERT: {e}")
            return False

    def unload_model(self):
        """Unload DrBERT model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        self.model_id = None
        gc.collect()

        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("DrBERT model unloaded")

    def _init_vector_store(self):
        """Initialize ChromaDB vector store."""
        if not CHROMADB_AVAILABLE:
            logger.warning("ChromaDB not available")
            return

        try:
            # Create persistent client
            os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
            self._chroma_client = chromadb.PersistentClient(
                path=CHROMA_PERSIST_DIR,
                settings=Settings(anonymized_telemetry=False)
            )

            # Get or create collections
            self._english_collection = self._chroma_client.get_or_create_collection(
                name=ENGLISH_PROTOCOLS_COLLECTION,
                metadata={"language": "en", "source": "MTS"}
            )

            self._french_collection = self._chroma_client.get_or_create_collection(
                name=FRENCH_PROTOCOLS_COLLECTION,
                metadata={"language": "fr", "source": "SFMU"}
            )

            self._case_collection = self._chroma_client.get_or_create_collection(
                name=PATIENT_CASES_COLLECTION,
                metadata={"type": "patient_cases"}
            )

            logger.info("ChromaDB vector store initialized")

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")

    def get_embeddings(self, texts: List[str], pooling: str = "mean") -> List[List[float]]:
        """
        Generate embeddings for texts using DrBERT.

        Args:
            texts: List of texts to embed
            pooling: Pooling strategy ('mean', 'cls', 'max')

        Returns:
            List of embedding vectors
        """
        if not self.is_loaded:
            raise RuntimeError("DrBERT model not loaded")

        embeddings = []

        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True,
                    padding=True
                ).to(self.device)

                outputs = self.model(**inputs)
                hidden_states = outputs.last_hidden_state

                if pooling == "cls":
                    embedding = hidden_states[:, 0, :]
                elif pooling == "max":
                    embedding = hidden_states.max(dim=1)[0]
                else:  # mean
                    attention_mask = inputs["attention_mask"].unsqueeze(-1)
                    embedding = (hidden_states * attention_mask).sum(1) / attention_mask.sum(1)

                embeddings.append(embedding.cpu().numpy().flatten().tolist())

        return embeddings

    def ingest_protocols(
        self,
        protocols: List[Dict[str, Any]],
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        Ingest protocols into the vector store.

        Args:
            protocols: List of protocol dicts with keys:
                       {id, title, text, source, category, keywords}
            language: 'en' or 'fr'

        Returns:
            Result dict with success status and counts
        """
        if not self.vector_store_available:
            return {"success": False, "error": "Vector store not available"}

        if not self.is_loaded:
            return {"success": False, "error": "DrBERT model not loaded"}

        collection = self._english_collection if language == "en" else self._french_collection
        added = 0
        skipped = 0

        try:
            for protocol in protocols:
                protocol_id = protocol.get("id", "")

                # Check if already exists
                existing = collection.get(ids=[protocol_id])
                if existing and existing.get("ids"):
                    skipped += 1
                    continue

                # Generate embedding from title + text
                embed_text = f"{protocol.get('title', '')} {protocol.get('text', '')}"
                embedding = self.get_embeddings([embed_text])[0]

                # Add to collection
                collection.add(
                    ids=[protocol_id],
                    embeddings=[embedding],
                    documents=[protocol.get("text", "")],
                    metadatas=[{
                        "title": protocol.get("title", ""),
                        "source": protocol.get("source", ""),
                        "category": protocol.get("category", ""),
                        "language": language,
                        "keywords": ",".join(protocol.get("keywords", [])),
                    }]
                )
                added += 1

            return {
                "success": True,
                "added": added,
                "skipped": skipped,
                "total": len(protocols),
            }

        except Exception as e:
            logger.error(f"Protocol ingestion error: {e}")
            return {"success": False, "error": str(e)}

    def search_protocols(
        self,
        query: str,
        language: Optional[str] = None,
        n_results: int = 3,
        category: Optional[str] = None
    ) -> SearchResult:
        """
        Search protocols using semantic similarity.

        Args:
            query: Search query
            language: 'en', 'fr', or None for auto-detect
            n_results: Number of results to return
            category: Optional category filter

        Returns:
            SearchResult with matching protocols
        """
        if not self.rag_available:
            return SearchResult(
                protocols=[],
                query_used=query,
                language=language or "en",
                total_found=0
            )

        # Auto-detect language if not specified
        if language is None:
            language = detect_language(query)

        collection = self._english_collection if language == "en" else self._french_collection

        try:
            # Generate query embedding
            query_embedding = self.get_embeddings([query])[0]

            # Build where clause if category filter
            where_clause = {"category": category} if category else None

            # Search
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )

            protocols = []
            if results and results.get("ids") and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
                    distance = results["distances"][0][i] if results.get("distances") else 1.0

                    # Convert distance to relevance score (lower distance = higher relevance)
                    relevance = max(0, 1 - distance / 2)

                    protocols.append(Protocol(
                        id=doc_id,
                        title=metadata.get("title", ""),
                        text=results["documents"][0][i] if results.get("documents") else "",
                        source=metadata.get("source", ""),
                        category=metadata.get("category", ""),
                        language=language,
                        relevance_score=relevance,
                        keywords=metadata.get("keywords", "").split(",") if metadata.get("keywords") else [],
                    ))

            return SearchResult(
                protocols=protocols,
                query_used=query,
                language=language,
                total_found=len(protocols)
            )

        except Exception as e:
            logger.error(f"Protocol search error: {e}")
            return SearchResult(
                protocols=[],
                query_used=query,
                language=language,
                total_found=0
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        stats = {
            "model_loaded": self.is_loaded,
            "model_id": self.model_id,
            "vector_store_available": self.vector_store_available,
            "device": str(self.device) if self.device else None,
        }

        if self.vector_store_available:
            try:
                stats["english_protocols"] = self._english_collection.count()
                stats["french_protocols"] = self._french_collection.count()
                stats["patient_cases"] = self._case_collection.count()
                stats["total_protocols"] = stats["english_protocols"] + stats["french_protocols"]
            except Exception as e:
                logger.error(f"Error getting stats: {e}")

        return stats

    def clear_protocols(self, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Clear protocols from vector store.

        Args:
            language: 'en', 'fr', or None for all

        Returns:
            Result dict
        """
        if not self.vector_store_available:
            return {"success": False, "error": "Vector store not available"}

        try:
            if language is None or language == "en":
                self._chroma_client.delete_collection(ENGLISH_PROTOCOLS_COLLECTION)
                self._english_collection = self._chroma_client.get_or_create_collection(
                    name=ENGLISH_PROTOCOLS_COLLECTION
                )

            if language is None or language == "fr":
                self._chroma_client.delete_collection(FRENCH_PROTOCOLS_COLLECTION)
                self._french_collection = self._chroma_client.get_or_create_collection(
                    name=FRENCH_PROTOCOLS_COLLECTION
                )

            return {"success": True, "message": f"Cleared protocols for language: {language or 'all'}"}

        except Exception as e:
            logger.error(f"Error clearing protocols: {e}")
            return {"success": False, "error": str(e)}


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_drbert_instance: Optional[DrBERTRAG] = None


def get_drbert_rag(reinitialize: bool = False) -> DrBERTRAG:
    """
    Get singleton DrBERTRAG instance.

    Args:
        reinitialize: Force create new instance

    Returns:
        DrBERTRAG instance
    """
    global _drbert_instance

    if _drbert_instance is None or reinitialize:
        _drbert_instance = DrBERTRAG()

    return _drbert_instance
