"""
DrBERT Engine - Dual-Language Medical BERT with Smart GPU/CPU Split + ChromaDB Vector Store

Loads DrBERT/PubMedBERT hybrid model with automatic memory distribution
between GPU and CPU for laptop compatibility.

DUAL-LANGUAGE RAG ARCHITECTURE:
- Supports both English and French medical protocols
- Uses DrBERT-PubMedBERT hybrid for bilingual embeddings
- Language detection routes queries to appropriate collection
- No translation needed - native language semantic search

Use cases:
- Medical entity extraction (NER)
- Symptom classification
- Bilingual biomedical embeddings
- Fill-mask for medical terms
- Protocol semantic search (RAG) in EN/FR
"""

import os
import gc
import re
import threading
import logging
import hashlib
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

# =============================================================================
# Lazy imports for optional dependencies
# =============================================================================

TRANSFORMERS_AVAILABLE = False
TORCH_AVAILABLE = False
CHROMADB_AVAILABLE = False
SENTENCE_TRANSFORMERS_AVAILABLE = False

def _check_dependencies():
    global TRANSFORMERS_AVAILABLE, TORCH_AVAILABLE, CHROMADB_AVAILABLE, SENTENCE_TRANSFORMERS_AVAILABLE
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
        CHROMADB_AVAILABLE = True
    except ImportError:
        pass
    try:
        from sentence_transformers import SentenceTransformer
        SENTENCE_TRANSFORMERS_AVAILABLE = True
    except ImportError:
        pass

_check_dependencies()


# =============================================================================
# Language Detection
# =============================================================================

# Keywords for language detection (medical context)
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

    Uses keyword matching optimized for medical terminology.
    Returns 'en' for English, 'fr' for French.

    Args:
        text: Medical text to analyze

    Returns:
        'en' or 'fr'
    """
    if not text:
        return "en"  # Default to English

    text_lower = text.lower()

    # Count keyword matches
    fr_count = sum(1 for kw in FRENCH_KEYWORDS if kw in text_lower)
    en_count = sum(1 for kw in ENGLISH_KEYWORDS if kw in text_lower)

    # Return language with more matches, default to English on tie
    return "fr" if fr_count > en_count else "en"


# =============================================================================
# DrBERT Model Configs
# =============================================================================

@dataclass
class DrBERTConfig:
    """Configuration for DrBERT model variants."""
    id: str
    name: str
    hf_model_id: str
    description: str
    training_data_gb: int
    approx_size_mb: int
    supports_medical_ner: bool = True

DRBERT_MODELS: Dict[str, DrBERTConfig] = {
    "drbert-4gb": DrBERTConfig(
        id="drbert-4gb",
        name="DrBERT 4GB",
        hf_model_id="Dr-BERT/DrBERT-4GB",
        description="French medical BERT trained on 4GB NACHOS corpus",
        training_data_gb=4,
        approx_size_mb=440,
    ),
    "drbert-7gb": DrBERTConfig(
        id="drbert-7gb",
        name="DrBERT 7GB",
        hf_model_id="Dr-BERT/DrBERT-7GB",
        description="French medical BERT trained on 7GB NACHOS corpus (recommended)",
        training_data_gb=7,
        approx_size_mb=440,
    ),
    "drbert-4gb-pubmed": DrBERTConfig(
        id="drbert-4gb-pubmed",
        name="DrBERT 4GB + PubMedBERT",
        hf_model_id="Dr-BERT/DrBERT-4GB-CP-PubMedBERT",
        description="DrBERT with continued pre-training from PubMedBERT",
        training_data_gb=4,
        approx_size_mb=440,
    ),
}

# =============================================================================
# GPU/CPU Memory Utilities
# =============================================================================

def get_gpu_memory_info() -> Tuple[int, int, bool]:
    """
    Get GPU memory info: (total_mb, free_mb, cuda_available).
    Returns (0, 0, False) if no GPU available.
    """
    if not TORCH_AVAILABLE:
        return 0, 0, False

    import torch
    if not torch.cuda.is_available():
        return 0, 0, False

    try:
        torch.cuda.init()
        total = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
        reserved = torch.cuda.memory_reserved(0) // (1024 * 1024)
        free = total - reserved
        return total, free, True
    except Exception as e:
        logger.warning(f"GPU memory check failed: {e}")
        return 0, 0, False


def compute_device_map(model_size_mb: int, gpu_free_mb: int) -> Dict[str, Any]:
    """
    Compute optimal device_map for model layers.

    Strategy:
    - If GPU has >500MB free: load embeddings + first N layers on GPU
    - Remaining layers on CPU
    - Always keep final layer on CPU for flexibility
    """
    if not TORCH_AVAILABLE:
        return {"": "cpu"}

    import torch

    if not torch.cuda.is_available() or gpu_free_mb < 200:
        return {"": "cpu"}

    # BERT has 12 layers, embeddings ~90MB, each layer ~25MB
    # We want to fit as many layers as possible on GPU

    embedding_size_mb = 90
    layer_size_mb = 25
    safety_margin_mb = 100

    available = gpu_free_mb - safety_margin_mb

    if available < embedding_size_mb:
        return {"": "cpu"}

    available -= embedding_size_mb
    gpu_layers = min(12, available // layer_size_mb)

    if gpu_layers >= 12:
        # Full model fits on GPU
        return {"": "cuda:0"}

    if gpu_layers <= 2:
        # Not worth the overhead, use CPU
        return {"": "cpu"}

    # Build layer-wise device map for partial GPU offload
    device_map = {
        "embeddings": "cuda:0",
        "encoder.layer.0": "cuda:0",
    }

    for i in range(1, gpu_layers):
        device_map[f"encoder.layer.{i}"] = "cuda:0"

    for i in range(gpu_layers, 12):
        device_map[f"encoder.layer.{i}"] = "cpu"

    device_map["pooler"] = "cpu"

    return device_map


# =============================================================================
# Protocol Data Structures
# =============================================================================

@dataclass
class Protocol:
    """A French medical protocol for triage."""
    id: str
    title: str
    text: str
    source: str = "SFMU/HAS"
    category: str = "general"
    priority_level: Optional[str] = None  # "1" to "5" or color
    keywords: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PatientCase:
    """A patient case for similar case lookup."""
    case_id: str
    session_id: str
    chief_complaint: str
    symptoms_text: str  # Flattened symptoms in French
    risk_band: str
    demographics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# ChromaDB Vector Store for Protocol Retrieval
# =============================================================================

class ProtocolVectorStore:
    """
    ChromaDB-backed vector store for medical protocols (English and French).

    Uses DrBERT-PubMedBERT embeddings for bilingual semantic search.
    Maintains separate collections for each language.
    """

    def __init__(self,
                 persist_directory: str,
                 embedding_function: Optional[Any] = None):
        """
        Initialize the dual-language vector store.

        Args:
            persist_directory: Path to store ChromaDB data
            embedding_function: Custom embedding function (uses DrBERT if None)
        """
        self.persist_directory = persist_directory
        self._embedding_fn = embedding_function
        self._client = None
        self._en_collection = None  # English protocols
        self._fr_collection = None  # French protocols
        self._lock = threading.RLock()

        self._initialize_store()

    def _initialize_store(self):
        """Initialize ChromaDB client and dual collections."""
        if not CHROMADB_AVAILABLE:
            logger.warning("ChromaDB not available. Protocol storage disabled.")
            return

        try:
            import chromadb
            from chromadb.config import Settings

            # Create persist directory if needed
            os.makedirs(self.persist_directory, exist_ok=True)

            # Initialize persistent client
            self._client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            # Create dual collections for English and French
            self._en_collection = self._client.get_or_create_collection(
                name="english_protocols",
                metadata={"description": "English medical protocols for triage", "language": "en"}
            )
            self._fr_collection = self._client.get_or_create_collection(
                name="french_protocols",
                metadata={"description": "French medical protocols for triage", "language": "fr"}
            )

            logger.info(f"VectorStore initialized: EN={self._en_collection.count()}, FR={self._fr_collection.count()} protocols")

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            self._client = None
            self._en_collection = None
            self._fr_collection = None

    def _get_collection(self, language: str = "en"):
        """Get the appropriate collection for language."""
        return self._en_collection if language == "en" else self._fr_collection

    @property
    def is_available(self) -> bool:
        """Check if vector store is available."""
        return self._en_collection is not None or self._fr_collection is not None

    def set_embedding_function(self, fn):
        """Set the embedding function (typically from DrBERTEngine)."""
        self._embedding_fn = fn

    def _generate_id(self, protocol: Protocol, language: str = "en") -> str:
        """Generate deterministic ID for a protocol."""
        content = f"{language}:{protocol.title}:{protocol.text[:200]}:{protocol.source}"
        return hashlib.md5(content.encode()).hexdigest()

    def ingest_protocols(self, protocols: List[Protocol], language: str = "en", batch_size: int = 32) -> Dict[str, Any]:
        """
        Ingest protocols into the appropriate language collection.

        Args:
            protocols: List of Protocol objects
            language: 'en' for English, 'fr' for French
            batch_size: Number of protocols to process at once

        Returns:
            Summary of ingestion results
        """
        if not self.is_available:
            return {"success": False, "error": "Vector store not available"}

        if not self._embedding_fn:
            return {"success": False, "error": "No embedding function set. Load DrBERT first."}

        collection = self._get_collection(language)
        if not collection:
            return {"success": False, "error": f"No collection for language: {language}"}

        with self._lock:
            added = 0
            skipped = 0
            errors = []

            for i in range(0, len(protocols), batch_size):
                batch = protocols[i:i + batch_size]

                try:
                    # Prepare batch data
                    ids = [self._generate_id(p, language) for p in batch]
                    documents = [p.text for p in batch]
                    metadatas = [
                        {
                            "title": p.title,
                            "source": p.source,
                            "category": p.category,
                            "priority_level": p.priority_level or "",
                            "keywords": ",".join(p.keywords),
                            "language": language,
                            **{k: str(v) for k, v in p.metadata.items()}
                        }
                        for p in batch
                    ]

                    # Generate embeddings using DrBERT-PubMedBERT
                    embeddings = self._embedding_fn(documents)

                    # Check for existing IDs
                    existing = set()
                    try:
                        result = collection.get(ids=ids)
                        existing = set(result["ids"]) if result["ids"] else set()
                    except Exception:
                        pass

                    # Filter out existing
                    new_ids = []
                    new_docs = []
                    new_metas = []
                    new_embeds = []

                    for j, pid in enumerate(ids):
                        if pid not in existing:
                            new_ids.append(pid)
                            new_docs.append(documents[j])
                            new_metas.append(metadatas[j])
                            new_embeds.append(embeddings[j])
                        else:
                            skipped += 1

                    # Add new protocols
                    if new_ids:
                        collection.add(
                            ids=new_ids,
                            documents=new_docs,
                            metadatas=new_metas,
                            embeddings=new_embeds
                        )
                        added += len(new_ids)

                except Exception as e:
                    errors.append(str(e))
                    logger.error(f"Batch ingestion error: {e}")

            return {
                "success": len(errors) == 0,
                "added": added,
                "skipped": skipped,
                "total": len(protocols),
                "language": language,
                "total_in_store": collection.count(),
                "errors": errors[:5] if errors else []
            }

    def search(self,
               query: str,
               language: str = "en",
               n_results: int = 3,
               category_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search protocols using semantic similarity with keyword boosting.

        Args:
            query: Query text (should match the language of the collection)
            language: 'en' for English, 'fr' for French
            n_results: Number of results to return
            category_filter: Optional category filter

        Returns:
            List of matching protocols with scores
        """
        if not self.is_available:
            return []

        if not self._embedding_fn:
            logger.warning("No embedding function. Cannot search.")
            return []

        collection = self._get_collection(language)
        if not collection:
            logger.warning(f"No collection for language: {language}")
            return []

        with self._lock:
            try:
                # Generate query embedding
                query_embedding = self._embedding_fn([query])[0]

                # Build where clause if filtering
                where = None
                if category_filter:
                    where = {"category": category_filter}

                # Fetch more results for re-ranking
                fetch_n = max(n_results * 2, 8)

                # Search in language-specific collection
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=fetch_n,
                    where=where,
                    include=["documents", "metadatas", "distances"]
                )

                # Format results with keyword boosting
                protocols = []
                query_lower = query.lower()

                if results and results["ids"] and results["ids"][0]:
                    for i, doc_id in enumerate(results["ids"][0]):
                        distance = results["distances"][0][i] if results["distances"] else 0.0
                        metadata = results["metadatas"][0][i] if results["metadatas"] else {}

                        # Base relevance from semantic similarity
                        # For normalized embeddings: L2 = sqrt(2 * (1 - cosine_sim))
                        base_relevance = max(0.0, min(1.0, 1.0 - (distance * distance / 4.0)))

                        # Keyword boost: check if query terms match protocol keywords
                        keywords = metadata.get("keywords", "").lower().split(",")
                        title = metadata.get("title", "").lower()
                        keyword_boost = 0.0

                        for keyword in keywords:
                            keyword = keyword.strip()
                            if keyword and keyword in query_lower:
                                keyword_boost += 0.05  # Boost for each matching keyword

                        # Title match boost - use word boundary matching to avoid
                        # partial matches like "chest" in "manchester"
                        title_words = set(re.findall(r'\b\w+\b', title))
                        query_words = query_lower.split()
                        for word in query_words:
                            if len(word) > 3 and word in title_words:
                                keyword_boost += 0.08  # Stronger boost for title match

                        # Cap boost at 0.3
                        keyword_boost = min(keyword_boost, 0.3)

                        # Combined score
                        final_score = min(1.0, base_relevance + keyword_boost)

                        protocols.append({
                            "id": doc_id,
                            "text": results["documents"][0][i] if results["documents"] else "",
                            "metadata": metadata,
                            "distance": distance,
                            "relevance_score": final_score,
                            "semantic_score": base_relevance,
                            "keyword_boost": keyword_boost,
                            "language": language
                        })

                # Re-rank by final score and return top n
                protocols.sort(key=lambda x: x["relevance_score"], reverse=True)
                return protocols[:n_results]

            except Exception as e:
                logger.error(f"Search error: {e}")
                return []

    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics for both languages."""
        if not self.is_available:
            return {"available": False}

        return {
            "available": True,
            "english_protocols": self._en_collection.count() if self._en_collection else 0,
            "french_protocols": self._fr_collection.count() if self._fr_collection else 0,
            "total_protocols": (
                (self._en_collection.count() if self._en_collection else 0) +
                (self._fr_collection.count() if self._fr_collection else 0)
            ),
            "persist_directory": self.persist_directory
        }

    def clear(self, language: Optional[str] = None) -> bool:
        """
        Clear protocols from the store.

        Args:
            language: 'en', 'fr', or None for both

        Returns:
            True if cleared successfully
        """
        if not self.is_available:
            return False

        with self._lock:
            try:
                if language is None or language == "en":
                    if self._en_collection:
                        self._client.delete_collection("english_protocols")
                        self._en_collection = self._client.create_collection(
                            name="english_protocols",
                            metadata={"description": "English medical protocols for triage", "language": "en"}
                        )

                if language is None or language == "fr":
                    if self._fr_collection:
                        self._client.delete_collection("french_protocols")
                        self._fr_collection = self._client.create_collection(
                            name="french_protocols",
                            metadata={"description": "French medical protocols for triage", "language": "fr"}
                        )

                return True
            except Exception as e:
                logger.error(f"Clear error: {e}")
                return False


class PatientCaseVectorStore:
    """
    ChromaDB-backed vector store for patient case histories.

    Used to find similar past cases for new patients.
    Stores symptom summaries in French for DrBERT semantic search.
    """

    def __init__(self,
                 persist_directory: str,
                 collection_name: str = "patient_cases",
                 embedding_function: Optional[Any] = None):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self._embedding_fn = embedding_function
        self._client = None
        self._collection = None
        self._lock = threading.RLock()

        self._initialize_store()

    def _initialize_store(self):
        """Initialize ChromaDB client and collection."""
        if not CHROMADB_AVAILABLE:
            logger.warning("ChromaDB not available. Patient case storage disabled.")
            return

        try:
            import chromadb
            from chromadb.config import Settings

            os.makedirs(self.persist_directory, exist_ok=True)

            self._client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Patient case histories for similar case lookup"}
            )

            logger.info(f"PatientCaseStore initialized: {self._collection.count()} cases")

        except Exception as e:
            logger.error(f"Failed to initialize PatientCaseStore: {e}")
            self._client = None
            self._collection = None

    @property
    def is_available(self) -> bool:
        return self._collection is not None

    def set_embedding_function(self, fn):
        self._embedding_fn = fn

    def add_case(self, case: PatientCase) -> bool:
        """Add a single patient case to the store."""
        if not self.is_available or not self._embedding_fn:
            return False

        with self._lock:
            try:
                # Check if already exists
                existing = self._collection.get(ids=[case.case_id])
                if existing and existing["ids"]:
                    return True  # Already indexed

                # Generate embedding
                embedding = self._embedding_fn([case.symptoms_text])[0]

                self._collection.add(
                    ids=[case.case_id],
                    documents=[case.symptoms_text],
                    embeddings=[embedding],
                    metadatas=[{
                        "session_id": case.session_id,
                        "chief_complaint": case.chief_complaint,
                        "risk_band": case.risk_band,
                        "age": str(case.demographics.get("age", "")),
                        "sex": case.demographics.get("sex", ""),
                        **{k: str(v) for k, v in case.metadata.items()}
                    }]
                )
                return True

            except Exception as e:
                logger.error(f"Failed to add case: {e}")
                return False

    def add_cases_batch(self, cases: List[PatientCase], batch_size: int = 32) -> Dict[str, Any]:
        """Add multiple patient cases in batch."""
        if not self.is_available:
            return {"success": False, "error": "Store not available"}

        if not self._embedding_fn:
            return {"success": False, "error": "No embedding function set"}

        with self._lock:
            added = 0
            skipped = 0
            errors = []

            for i in range(0, len(cases), batch_size):
                batch = cases[i:i + batch_size]

                try:
                    # Check existing
                    ids = [c.case_id for c in batch]
                    existing = set()
                    try:
                        result = self._collection.get(ids=ids)
                        existing = set(result["ids"]) if result["ids"] else set()
                    except Exception:
                        pass

                    # Filter new cases
                    new_cases = [c for c in batch if c.case_id not in existing]
                    skipped += len(batch) - len(new_cases)

                    if not new_cases:
                        continue

                    # Generate embeddings
                    texts = [c.symptoms_text for c in new_cases]
                    embeddings = self._embedding_fn(texts)

                    # Add to collection
                    self._collection.add(
                        ids=[c.case_id for c in new_cases],
                        documents=texts,
                        embeddings=embeddings,
                        metadatas=[{
                            "session_id": c.session_id,
                            "chief_complaint": c.chief_complaint,
                            "risk_band": c.risk_band,
                            "age": str(c.demographics.get("age", "")),
                            "sex": c.demographics.get("sex", "")
                        } for c in new_cases]
                    )
                    added += len(new_cases)

                except Exception as e:
                    errors.append(str(e))
                    logger.error(f"Batch add error: {e}")

            return {
                "success": len(errors) == 0,
                "added": added,
                "skipped": skipped,
                "total_in_store": self._collection.count(),
                "errors": errors[:5]
            }

    def find_similar_cases(self,
                           symptoms_french: str,
                           n_results: int = 5,
                           risk_band_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Find similar past cases based on symptoms."""
        if not self.is_available or not self._embedding_fn:
            return []

        with self._lock:
            try:
                query_embedding = self._embedding_fn([symptoms_french])[0]

                where = None
                if risk_band_filter:
                    where = {"risk_band": risk_band_filter}

                results = self._collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    where=where,
                    include=["documents", "metadatas", "distances"]
                )

                cases = []
                if results and results["ids"] and results["ids"][0]:
                    for i, case_id in enumerate(results["ids"][0]):
                        cases.append({
                            "case_id": case_id,
                            "symptoms_text": results["documents"][0][i] if results["documents"] else "",
                            "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                            "distance": results["distances"][0][i] if results["distances"] else 0.0,
                            "similarity": 1.0 - (results["distances"][0][i] / 2.0) if results["distances"] else 0.5
                        })

                return cases

            except Exception as e:
                logger.error(f"Similar case search error: {e}")
                return []

    def get_stats(self) -> Dict[str, Any]:
        if not self.is_available:
            return {"available": False}

        return {
            "available": True,
            "collection_name": self.collection_name,
            "total_cases": self._collection.count(),
            "persist_directory": self.persist_directory
        }

    def clear(self) -> bool:
        if not self.is_available:
            return False

        with self._lock:
            try:
                self._client.delete_collection(self.collection_name)
                self._collection = self._client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Patient case histories for similar case lookup"}
                )
                return True
            except Exception as e:
                logger.error(f"Clear error: {e}")
                return False


# =============================================================================
# DrBERT Engine
# =============================================================================

@dataclass
class LoadedDrBERT:
    """Container for loaded DrBERT model."""
    config: DrBERTConfig
    model: Any
    tokenizer: Any
    device_map: Dict[str, Any]
    loaded_at: datetime
    inference_count: int = 0


class DrBERTEngine:
    """
    DrBERT engine with smart GPU/CPU memory distribution.

    Features:
    - Auto-detects GPU VRAM and splits model accordingly
    - Thread-safe loading/unloading
    - Medical text embeddings
    - Fill-mask for French medical terms
    - Entity extraction (with fine-tuned models)
    - ChromaDB vector store for protocol retrieval (RAG)
    """

    # Default model: DrBERT-PubMedBERT hybrid works for both EN and FR
    DEFAULT_MODEL_ID = "drbert-4gb-pubmed"

    def __init__(self,
                 cache_dir: Optional[str] = None,
                 vector_db_dir: Optional[str] = None,
                 auto_load: bool = False):
        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(__file__), "..", "models", "drbert_cache"
        )
        self.vector_db_dir = vector_db_dir or os.path.join(
            os.path.dirname(__file__), "..", "data", "vector_db"
        )
        # Protocol files for both languages
        self.en_protocols_file = os.path.join(
            os.path.dirname(__file__), "config", "english_protocols.json"
        )
        self.fr_protocols_file = os.path.join(
            os.path.dirname(__file__), "config", "french_protocols.json"
        )
        self._current_model: Optional[LoadedDrBERT] = None
        self._lock = threading.RLock()
        self._total_inferences = 0

        # Initialize vector stores (dual-language)
        self._vector_store: Optional[ProtocolVectorStore] = None
        self._case_store: Optional[PatientCaseVectorStore] = None
        self._init_vector_stores()

        # Callback for auto-ingestion (set by main.py)
        self._on_load_callback: Optional[Any] = None

        if auto_load and TRANSFORMERS_AVAILABLE:
            self._try_load_default()

    def _init_vector_stores(self):
        """Initialize dual-language protocol and patient case vector stores."""
        if CHROMADB_AVAILABLE:
            try:
                self._vector_store = ProtocolVectorStore(
                    persist_directory=self.vector_db_dir
                )
                self._case_store = PatientCaseVectorStore(
                    persist_directory=self.vector_db_dir,
                    collection_name="patient_cases"
                )
                logger.info(f"VectorStores initialized at {self.vector_db_dir}")
            except Exception as e:
                logger.error(f"Failed to initialize VectorStores: {e}")
                self._vector_store = None
                self._case_store = None
        else:
            logger.warning("ChromaDB not available. Vector stores disabled.")

    def set_on_load_callback(self, callback):
        """Set callback to run after model loads (for auto-ingestion)."""
        self._on_load_callback = callback

    def _try_load_default(self):
        """Attempt to load the recommended DrBERT-PubMedBERT hybrid model."""
        self.load_model(self.DEFAULT_MODEL_ID)

    # =========================================================================
    # Model Loading
    # =========================================================================

    def load_model(self, model_id: str = "drbert-7gb") -> bool:
        """
        Load DrBERT model with automatic GPU/CPU distribution.

        Args:
            model_id: One of 'drbert-4gb', 'drbert-7gb', 'drbert-4gb-pubmed'

        Returns:
            True if loaded successfully
        """
        if not TRANSFORMERS_AVAILABLE:
            logger.error("transformers not installed. Run: pip install transformers accelerate")
            return False

        config = DRBERT_MODELS.get(model_id)
        if not config:
            logger.error(f"Unknown DrBERT model: {model_id}")
            return False

        with self._lock:
            # Already loaded?
            if self._current_model and self._current_model.config.id == model_id:
                return True

            # Unload existing
            self.unload_model()

            try:
                from transformers import AutoModel, AutoTokenizer
                import torch

                print(f"Loading {config.name}...")

                # Check GPU memory
                total_mb, free_mb, cuda_avail = get_gpu_memory_info()
                if cuda_avail:
                    print(f"  GPU detected: {free_mb}MB free / {total_mb}MB total")
                else:
                    print("  No GPU detected, using CPU")

                # Compute device map
                device_map = compute_device_map(config.approx_size_mb, free_mb)

                # Log device distribution
                if device_map.get("") == "cuda:0":
                    print("  Loading full model on GPU")
                elif device_map.get("") == "cpu":
                    print("  Loading full model on CPU")
                else:
                    gpu_layers = sum(1 for k, v in device_map.items() if "cuda" in str(v))
                    cpu_layers = sum(1 for k, v in device_map.items() if v == "cpu")
                    print(f"  Hybrid load: {gpu_layers} components on GPU, {cpu_layers} on CPU")

                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    config.hf_model_id,
                    cache_dir=self.cache_dir
                )

                # Load model with device map
                if device_map.get("") in ["cpu", "cuda:0"]:
                    # Simple case: all on one device
                    device = device_map.get("")
                    model = AutoModel.from_pretrained(
                        config.hf_model_id,
                        cache_dir=self.cache_dir
                    )
                    if device == "cuda:0":
                        model = model.cuda()
                    model.eval()
                else:
                    # Complex case: use accelerate for layer distribution
                    try:
                        from accelerate import dispatch_model, infer_auto_device_map

                        # Load to CPU first, then dispatch
                        model = AutoModel.from_pretrained(
                            config.hf_model_id,
                            cache_dir=self.cache_dir
                        )
                        model = dispatch_model(model, device_map=device_map)
                        model.eval()
                    except ImportError:
                        # Fallback: load on CPU if accelerate not available
                        logger.warning("accelerate not installed, falling back to CPU")
                        model = AutoModel.from_pretrained(
                            config.hf_model_id,
                            cache_dir=self.cache_dir
                        )
                        model.eval()
                        device_map = {"": "cpu"}

                self._current_model = LoadedDrBERT(
                    config=config,
                    model=model,
                    tokenizer=tokenizer,
                    device_map=device_map,
                    loaded_at=datetime.now()
                )

                # Link embedding function to vector stores
                if self._vector_store:
                    self._vector_store.set_embedding_function(self.get_embeddings)
                if self._case_store:
                    self._case_store.set_embedding_function(self.get_embeddings)

                if self._vector_store or self._case_store:
                    print(f"  VectorStores linked to DrBERT embeddings")

                # Auto-load protocols from file if available
                self._auto_load_protocols()

                print(f"  {config.name} loaded successfully!")

                # Run callback for auto-ingestion of patient cases
                if self._on_load_callback:
                    try:
                        self._on_load_callback()
                    except Exception as e:
                        logger.error(f"On-load callback error: {e}")

                return True

            except Exception as e:
                logger.error(f"Failed to load DrBERT: {e}")
                print(f"  Error loading {config.name}: {e}")
                return False

    def unload_model(self):
        """Unload current model and free memory."""
        with self._lock:
            if self._current_model:
                try:
                    del self._current_model.model
                    del self._current_model.tokenizer
                    if TORCH_AVAILABLE:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    gc.collect()
                except Exception as e:
                    logger.warning(f"Error during unload: {e}")
                self._current_model = None

    @property
    def is_loaded(self) -> bool:
        return self._current_model is not None

    def get_current_model(self) -> Optional[Dict[str, Any]]:
        """Get info about currently loaded model."""
        with self._lock:
            if not self._current_model:
                return None

            device_str = "CPU"
            dm = self._current_model.device_map
            if dm.get("") == "cuda:0":
                device_str = "GPU (full)"
            elif dm.get("") != "cpu" and dm:
                gpu_count = sum(1 for v in dm.values() if "cuda" in str(v))
                device_str = f"Hybrid ({gpu_count} on GPU)"

            return {
                "model_id": self._current_model.config.id,
                "name": self._current_model.config.name,
                "hf_model_id": self._current_model.config.hf_model_id,
                "device": device_str,
                "loaded_at": self._current_model.loaded_at.isoformat(),
                "inference_count": self._current_model.inference_count
            }

    # =========================================================================
    # Inference Methods
    # =========================================================================

    def get_embeddings(self, texts: List[str], pooling: str = "mean", normalize: bool = True) -> List[List[float]]:
        """
        Get embeddings for texts using DrBERT.

        Args:
            texts: List of medical texts (English or French)
            pooling: 'mean', 'cls', or 'max'
            normalize: Whether to L2-normalize embeddings (required for proper similarity scores)

        Returns:
            List of embedding vectors (768-dim each)
        """
        if not self._current_model:
            raise RuntimeError("No DrBERT model loaded")

        import torch

        with self._lock:
            model = self._current_model.model
            tokenizer = self._current_model.tokenizer

            # Tokenize
            inputs = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            # Move to appropriate device
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                hidden_states = outputs.last_hidden_state
                attention_mask = inputs["attention_mask"]

                if pooling == "cls":
                    embeddings = hidden_states[:, 0, :]
                elif pooling == "max":
                    # Mask padding tokens
                    mask = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                    hidden_states[mask == 0] = -1e9
                    embeddings = torch.max(hidden_states, dim=1)[0]
                else:  # mean pooling
                    mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                    sum_embeddings = torch.sum(hidden_states * mask, dim=1)
                    sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
                    embeddings = sum_embeddings / sum_mask

                # L2 normalize embeddings for proper cosine similarity via L2 distance
                if normalize:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                self._current_model.inference_count += len(texts)
                self._total_inferences += len(texts)

                return embeddings.cpu().tolist()

    def fill_mask(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Fill masked token in French medical text.

        Args:
            text: Text with <mask> token (e.g., "Le patient souffre de <mask>")
            top_k: Number of predictions to return

        Returns:
            List of predictions with scores
        """
        if not self._current_model:
            raise RuntimeError("No DrBERT model loaded")

        import torch

        with self._lock:
            model = self._current_model.model
            tokenizer = self._current_model.tokenizer

            # Ensure <mask> is in the text
            if "<mask>" not in text.lower():
                raise ValueError("Text must contain <mask> token")

            # Replace with model's mask token
            text = text.replace("<mask>", tokenizer.mask_token)
            text = text.replace("<MASK>", tokenizer.mask_token)

            inputs = tokenizer(text, return_tensors="pt")
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Find mask position
            mask_idx = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

            if len(mask_idx) == 0:
                raise ValueError("Mask token not found in tokenized input")

            with torch.no_grad():
                outputs = model(**inputs)

                # Get predictions at mask position
                # Note: For fill-mask, we need the MLM head, but base model doesn't have it
                # We'll use the hidden states and find nearest tokens
                hidden = outputs.last_hidden_state[0, mask_idx[0], :]

                # Get token embeddings
                token_embeddings = model.embeddings.word_embeddings.weight

                # Compute similarity
                similarities = torch.matmul(token_embeddings, hidden)
                top_indices = torch.topk(similarities, top_k).indices

                self._current_model.inference_count += 1
                self._total_inferences += 1

                results = []
                for idx in top_indices:
                    token = tokenizer.decode([idx])
                    score = similarities[idx].item()
                    filled = text.replace(tokenizer.mask_token, token)
                    results.append({
                        "token": token.strip(),
                        "score": score,
                        "sequence": filled
                    })

                return results

    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between two French medical texts.

        Returns:
            Cosine similarity score (0-1)
        """
        embeddings = self.get_embeddings([text1, text2])

        import math

        def cosine_sim(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            return dot / (norm_a * norm_b) if norm_a and norm_b else 0

        return cosine_sim(embeddings[0], embeddings[1])

    # =========================================================================
    # Auto-Load Protocols from File
    # =========================================================================

    def _auto_load_protocols(self):
        """Auto-load protocols from JSON files for both languages."""
        if not self._vector_store or not self._vector_store.is_available:
            return

        stats = self._vector_store.get_stats()

        # Load English protocols
        en_count = stats.get("english_protocols", 0)
        if en_count == 0 and os.path.exists(self.en_protocols_file):
            try:
                with open(self.en_protocols_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                protocols = data.get("protocols", [])
                if protocols:
                    print(f"  Auto-loading {len(protocols)} English protocols...")
                    result = self.ingest_protocols(protocols, language="en")
                    if result.get("success"):
                        print(f"  Loaded {result.get('added', 0)} English protocols")
                    else:
                        print(f"  EN protocol load failed: {result.get('error', 'Unknown')}")
            except Exception as e:
                logger.error(f"Failed to auto-load English protocols: {e}")
        elif en_count > 0:
            print(f"  English protocols already indexed: {en_count}")

        # Load French protocols
        fr_count = stats.get("french_protocols", 0)
        if fr_count == 0 and os.path.exists(self.fr_protocols_file):
            try:
                with open(self.fr_protocols_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                protocols = data.get("protocols", [])
                if protocols:
                    print(f"  Auto-loading {len(protocols)} French protocols...")
                    result = self.ingest_protocols(protocols, language="fr")
                    if result.get("success"):
                        print(f"  Loaded {result.get('added', 0)} French protocols")
                    else:
                        print(f"  FR protocol load failed: {result.get('error', 'Unknown')}")
            except Exception as e:
                logger.error(f"Failed to auto-load French protocols: {e}")
        elif fr_count > 0:
            print(f"  French protocols already indexed: {fr_count}")

    # =========================================================================
    # Protocol RAG Methods (Dual-Language Architecture)
    # =========================================================================

    def ingest_protocols(self,
                         protocols: List[Dict[str, Any]],
                         language: str = "en",
                         batch_size: int = 32) -> Dict[str, Any]:
        """
        Ingest medical protocols into the appropriate language collection.

        Args:
            protocols: List of protocol dicts with keys:
                - title: Protocol title
                - text: Full protocol text
                - source: Source (e.g., "SFMU", "HAS")
                - category: Category (e.g., "cardiac", "trauma")
                - priority_level: Optional priority level
                - keywords: Optional list of keywords
            language: 'en' for English, 'fr' for French
            batch_size: Batch size for embedding generation

        Returns:
            Ingestion result summary
        """
        if not self._vector_store:
            return {"success": False, "error": "Vector store not available"}

        if not self._current_model:
            return {"success": False, "error": "DrBERT model not loaded. Load model first."}

        # Convert dicts to Protocol objects
        protocol_objs = []
        for p in protocols:
            protocol_objs.append(Protocol(
                id=p.get("id", ""),
                title=p.get("title", "Untitled"),
                text=p.get("text", ""),
                source=p.get("source", "SFMU/HAS"),
                category=p.get("category", "general"),
                priority_level=p.get("priority_level"),
                keywords=p.get("keywords", []),
                metadata=p.get("metadata", {})
            ))

        return self._vector_store.ingest_protocols(protocol_objs, language=language, batch_size=batch_size)

    def search_protocols(self,
                         query: str,
                         language: Optional[str] = None,
                         n_results: int = 3,
                         category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search protocols using native language query.

        Auto-detects language if not specified, then searches the
        appropriate language collection.

        Args:
            query: Query text (English or French)
            language: 'en', 'fr', or None for auto-detect
            n_results: Number of results to return
            category: Optional category filter

        Returns:
            List of matching protocols with relevance scores
        """
        if not self._vector_store:
            logger.warning("Vector store not available for search")
            return []

        if not self._current_model:
            logger.warning("DrBERT not loaded. Cannot search protocols.")
            return []

        # Auto-detect language if not specified
        if language is None:
            language = detect_language(query)
            logger.debug(f"Auto-detected language: {language} for query: {query[:50]}...")

        return self._vector_store.search(
            query=query,
            language=language,
            n_results=n_results,
            category_filter=category
        )

    def get_best_protocol(self, query: str, language: Optional[str] = None, category: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get the single best matching protocol for a query.

        Args:
            query: Query text (English or French)
            language: 'en', 'fr', or None for auto-detect
            category: Optional category filter

        Returns:
            Best matching protocol or None
        """
        results = self.search_protocols(query, language=language, n_results=1, category=category)
        return results[0] if results else None

    def get_vector_store_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        if not self._vector_store:
            return {"available": False, "error": "Vector store not initialized"}
        return self._vector_store.get_stats()

    def clear_protocols(self) -> bool:
        """Clear all protocols from vector store."""
        if not self._vector_store:
            return False
        return self._vector_store.clear()

    @property
    def vector_store_available(self) -> bool:
        """Check if vector store is available for RAG."""
        return self._vector_store is not None and self._vector_store.is_available

    @property
    def rag_ready(self) -> bool:
        """Check if RAG pipeline is ready (model loaded + vector store available)."""
        return self.is_loaded and self.vector_store_available

    # =========================================================================
    # Patient Case Indexing Methods
    # =========================================================================

    def index_patient_case(self,
                           case_id: str,
                           session_id: str,
                           chief_complaint: str,
                           symptoms_text: str,
                           risk_band: str,
                           demographics: Dict[str, Any] = None) -> bool:
        """
        Index a single patient case for similar case lookup.

        Args:
            case_id: Unique case identifier
            session_id: Session ID
            chief_complaint: Main complaint
            symptoms_text: Symptoms summary (ideally in French)
            risk_band: Risk level (red/amber/green)
            demographics: Patient demographics

        Returns:
            True if indexed successfully
        """
        if not self._case_store:
            return False

        case = PatientCase(
            case_id=case_id,
            session_id=session_id,
            chief_complaint=chief_complaint,
            symptoms_text=symptoms_text,
            risk_band=risk_band,
            demographics=demographics or {}
        )

        return self._case_store.add_case(case)

    def index_patient_cases_batch(self,
                                   cases: List[Dict[str, Any]],
                                   batch_size: int = 32) -> Dict[str, Any]:
        """
        Index multiple patient cases in batch.

        Args:
            cases: List of case dicts with keys:
                - case_id, session_id, chief_complaint
                - symptoms_text (French preferred)
                - risk_band, demographics
            batch_size: Batch size for processing

        Returns:
            Result summary
        """
        if not self._case_store:
            return {"success": False, "error": "Case store not available"}

        if not self._current_model:
            return {"success": False, "error": "DrBERT model not loaded"}

        case_objs = []
        for c in cases:
            case_objs.append(PatientCase(
                case_id=c.get("case_id", ""),
                session_id=c.get("session_id", ""),
                chief_complaint=c.get("chief_complaint", ""),
                symptoms_text=c.get("symptoms_text", ""),
                risk_band=c.get("risk_band", ""),
                demographics=c.get("demographics", {}),
                metadata=c.get("metadata", {})
            ))

        return self._case_store.add_cases_batch(case_objs, batch_size)

    def find_similar_cases(self,
                           symptoms_french: str,
                           n_results: int = 5,
                           risk_band: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Find similar past patient cases.

        Args:
            symptoms_french: Symptoms in French (use LLM pivot if needed)
            n_results: Number of results
            risk_band: Optional filter by risk band

        Returns:
            List of similar cases with similarity scores
        """
        if not self._case_store or not self._current_model:
            return []

        return self._case_store.find_similar_cases(
            symptoms_french=symptoms_french,
            n_results=n_results,
            risk_band_filter=risk_band
        )

    def get_case_store_stats(self) -> Dict[str, Any]:
        """Get patient case store statistics."""
        if not self._case_store:
            return {"available": False}
        return self._case_store.get_stats()

    def clear_patient_cases(self) -> bool:
        """Clear all patient cases from store."""
        if not self._case_store:
            return False
        return self._case_store.clear()

    @property
    def case_store_available(self) -> bool:
        """Check if patient case store is available."""
        return self._case_store is not None and self._case_store.is_available

    # =========================================================================
    # Status & Info
    # =========================================================================

    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available DrBERT model variants."""
        return [
            {
                "id": cfg.id,
                "name": cfg.name,
                "hf_model_id": cfg.hf_model_id,
                "description": cfg.description,
                "training_data_gb": cfg.training_data_gb,
                "approx_size_mb": cfg.approx_size_mb,
                "is_loaded": (self._current_model and self._current_model.config.id == cfg.id)
            }
            for cfg in DRBERT_MODELS.values()
        ]

    def get_engine_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        total_mb, free_mb, cuda_avail = get_gpu_memory_info()

        stats = {
            "transformers_available": TRANSFORMERS_AVAILABLE,
            "torch_available": TORCH_AVAILABLE,
            "chromadb_available": CHROMADB_AVAILABLE,
            "cuda_available": cuda_avail,
            "gpu_total_mb": total_mb,
            "gpu_free_mb": free_mb,
            "model_loaded": self._current_model.config.name if self._current_model else None,
            "total_inferences": self._total_inferences,
            "rag_ready": self.rag_ready,
        }

        # Add protocol vector store stats
        if self._vector_store:
            stats["vector_store"] = self._vector_store.get_stats()
        else:
            stats["vector_store"] = {"available": False}

        # Add patient case store stats
        if self._case_store:
            stats["case_store"] = self._case_store.get_stats()
        else:
            stats["case_store"] = {"available": False}

        return stats


# =============================================================================
# Singleton
# =============================================================================

_drbert_instance: Optional[DrBERTEngine] = None


def get_drbert_engine(
    cache_dir: Optional[str] = None,
    vector_db_dir: Optional[str] = None,
    reinitialize: bool = False
) -> DrBERTEngine:
    """Get or create DrBERT engine singleton."""
    global _drbert_instance

    if _drbert_instance is None or reinitialize:
        if _drbert_instance and reinitialize:
            _drbert_instance.unload_model()
        _drbert_instance = DrBERTEngine(
            cache_dir=cache_dir,
            vector_db_dir=vector_db_dir,
            auto_load=False
        )

    return _drbert_instance


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "DrBERTEngine",
    "DrBERTConfig",
    "DRBERT_MODELS",
    "Protocol",
    "PatientCase",
    "ProtocolVectorStore",
    "PatientCaseVectorStore",
    "get_drbert_engine",
    "detect_language",
    "TRANSFORMERS_AVAILABLE",
    "CHROMADB_AVAILABLE",
    "FRENCH_KEYWORDS",
    "ENGLISH_KEYWORDS",
]
