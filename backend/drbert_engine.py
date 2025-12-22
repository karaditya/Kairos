"""
DrBERT Engine - French Medical BERT with Smart GPU/CPU Split + ChromaDB Vector Store

Loads DrBERT models (4GB/7GB training data variants) with automatic
memory distribution between GPU and CPU for laptop compatibility.

FRENCH PIVOT RAG ARCHITECTURE:
- Ingests French medical protocols into ChromaDB vector store
- Provides semantic search over protocols using DrBERT embeddings
- Enables multilingual queries via LLM translation to French

Use cases:
- Medical entity extraction (NER)
- Symptom classification
- French biomedical embeddings
- Fill-mask for medical terms
- Protocol semantic search (RAG)
"""

import os
import gc
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


# =============================================================================
# ChromaDB Vector Store for Protocol Retrieval
# =============================================================================

class ProtocolVectorStore:
    """
    ChromaDB-backed vector store for French medical protocols.

    Uses DrBERT embeddings for semantic search over protocol texts.
    Designed for the French Pivot RAG architecture.
    """

    def __init__(self,
                 persist_directory: str,
                 collection_name: str = "french_protocols",
                 embedding_function: Optional[Any] = None):
        """
        Initialize the vector store.

        Args:
            persist_directory: Path to store ChromaDB data
            collection_name: Name of the collection
            embedding_function: Custom embedding function (uses DrBERT if None)
        """
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

            # Get or create collection
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "French medical protocols for triage"}
            )

            logger.info(f"VectorStore initialized: {self._collection.count()} protocols")

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            self._client = None
            self._collection = None

    @property
    def is_available(self) -> bool:
        """Check if vector store is available."""
        return self._collection is not None

    def set_embedding_function(self, fn):
        """Set the embedding function (typically from DrBERTEngine)."""
        self._embedding_fn = fn

    def _generate_id(self, protocol: Protocol) -> str:
        """Generate deterministic ID for a protocol."""
        content = f"{protocol.title}:{protocol.text[:200]}:{protocol.source}"
        return hashlib.md5(content.encode()).hexdigest()

    def ingest_protocols(self, protocols: List[Protocol], batch_size: int = 32) -> Dict[str, Any]:
        """
        Ingest protocols into the vector store.

        Args:
            protocols: List of Protocol objects
            batch_size: Number of protocols to process at once

        Returns:
            Summary of ingestion results
        """
        if not self.is_available:
            return {"success": False, "error": "Vector store not available"}

        if not self._embedding_fn:
            return {"success": False, "error": "No embedding function set. Load DrBERT first."}

        with self._lock:
            added = 0
            skipped = 0
            errors = []

            for i in range(0, len(protocols), batch_size):
                batch = protocols[i:i + batch_size]

                try:
                    # Prepare batch data
                    ids = [self._generate_id(p) for p in batch]
                    documents = [p.text for p in batch]
                    metadatas = [
                        {
                            "title": p.title,
                            "source": p.source,
                            "category": p.category,
                            "priority_level": p.priority_level or "",
                            "keywords": ",".join(p.keywords),
                            **{k: str(v) for k, v in p.metadata.items()}
                        }
                        for p in batch
                    ]

                    # Generate embeddings using DrBERT
                    embeddings = self._embedding_fn(documents)

                    # Check for existing IDs
                    existing = set()
                    try:
                        result = self._collection.get(ids=ids)
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
                        self._collection.add(
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
                "total_in_store": self._collection.count(),
                "errors": errors[:5] if errors else []
            }

    def search(self,
               query: str,
               n_results: int = 3,
               category_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search protocols using semantic similarity.

        Args:
            query: French query text (should be in French for best results!)
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

        with self._lock:
            try:
                # Generate query embedding
                query_embedding = self._embedding_fn([query])[0]

                # Build where clause if filtering
                where = None
                if category_filter:
                    where = {"category": category_filter}

                # Search
                results = self._collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    where=where,
                    include=["documents", "metadatas", "distances"]
                )

                # Format results
                protocols = []
                if results and results["ids"] and results["ids"][0]:
                    for i, doc_id in enumerate(results["ids"][0]):
                        protocols.append({
                            "id": doc_id,
                            "text": results["documents"][0][i] if results["documents"] else "",
                            "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                            "distance": results["distances"][0][i] if results["distances"] else 0.0,
                            "relevance_score": 1.0 - (results["distances"][0][i] / 2.0) if results["distances"] else 0.5
                        })

                return protocols

            except Exception as e:
                logger.error(f"Search error: {e}")
                return []

    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        if not self.is_available:
            return {"available": False}

        return {
            "available": True,
            "collection_name": self.collection_name,
            "total_protocols": self._collection.count(),
            "persist_directory": self.persist_directory
        }

    def clear(self) -> bool:
        """Clear all protocols from the store."""
        if not self.is_available:
            return False

        with self._lock:
            try:
                # Delete and recreate collection
                self._client.delete_collection(self.collection_name)
                self._collection = self._client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "French medical protocols for triage"}
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
        self._current_model: Optional[LoadedDrBERT] = None
        self._lock = threading.RLock()
        self._total_inferences = 0

        # Initialize vector store
        self._vector_store: Optional[ProtocolVectorStore] = None
        self._init_vector_store()

        if auto_load and TRANSFORMERS_AVAILABLE:
            self._try_load_default()

    def _init_vector_store(self):
        """Initialize the protocol vector store."""
        if CHROMADB_AVAILABLE:
            try:
                self._vector_store = ProtocolVectorStore(
                    persist_directory=self.vector_db_dir,
                    collection_name="french_protocols"
                )
                logger.info(f"VectorStore initialized at {self.vector_db_dir}")
            except Exception as e:
                logger.error(f"Failed to initialize VectorStore: {e}")
                self._vector_store = None
        else:
            logger.warning("ChromaDB not available. Vector store disabled.")

    def _try_load_default(self):
        """Attempt to load the recommended DrBERT model."""
        self.load_model("drbert-7gb")

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

                # Link embedding function to vector store
                if self._vector_store:
                    self._vector_store.set_embedding_function(self.get_embeddings)
                    print(f"  VectorStore linked to DrBERT embeddings")

                print(f"  {config.name} loaded successfully!")
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

    def get_embeddings(self, texts: List[str], pooling: str = "mean") -> List[List[float]]:
        """
        Get embeddings for texts using DrBERT.

        Args:
            texts: List of French medical texts
            pooling: 'mean', 'cls', or 'max'

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
    # Protocol RAG Methods (French Pivot Architecture)
    # =========================================================================

    def ingest_protocols(self,
                         protocols: List[Dict[str, Any]],
                         batch_size: int = 32) -> Dict[str, Any]:
        """
        Ingest French medical protocols into the vector store.

        Args:
            protocols: List of protocol dicts with keys:
                - title: Protocol title
                - text: Full protocol text (in French)
                - source: Source (e.g., "SFMU", "HAS")
                - category: Category (e.g., "cardiac", "trauma")
                - priority_level: Optional priority level
                - keywords: Optional list of keywords
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

        return self._vector_store.ingest_protocols(protocol_objs, batch_size)

    def search_protocols(self,
                         french_query: str,
                         n_results: int = 3,
                         category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search protocols using French query text.

        IMPORTANT: Query MUST be in French for best results!
        Use the LLM to translate user input to French before calling this.

        Args:
            french_query: Query in French (e.g., "Douleur thoracique constrictive")
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

        return self._vector_store.search(
            query=french_query,
            n_results=n_results,
            category_filter=category
        )

    def get_best_protocol(self, french_query: str, category: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get the single best matching protocol for a French query.

        Args:
            french_query: Query in French
            category: Optional category filter

        Returns:
            Best matching protocol or None
        """
        results = self.search_protocols(french_query, n_results=1, category=category)
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

        # Add vector store stats
        if self._vector_store:
            vs_stats = self._vector_store.get_stats()
            stats["vector_store"] = vs_stats
        else:
            stats["vector_store"] = {"available": False}

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
    "ProtocolVectorStore",
    "get_drbert_engine",
    "TRANSFORMERS_AVAILABLE",
    "CHROMADB_AVAILABLE",
]
