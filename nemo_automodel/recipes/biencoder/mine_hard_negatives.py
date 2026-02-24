# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hard negative mining recipe for biencoder models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from nemo_automodel._transformers.auto_model import NeMoAutoModelBiencoder
from nemo_automodel._transformers.auto_tokenizer import NeMoAutoTokenizer
from nemo_automodel.components.datasets.llm.retrieval_dataset import load_datasets
from nemo_automodel.components.distributed.init_utils import DistInfo, initialize_distributed

logger = logging.getLogger(__name__)

# Placeholder for empty questions to preserve index alignment
EMPTY_QUESTION = "##### keep empty questions #####"

# Cache file names
QUERY_EMBEDDINGS_FNAME = "query_embeddings.npz"
DOCUMENT_EMBEDDINGS_FNAME = "passage_embeddings.npz"
CORPUS_CHUNKS_DIR = "corpus_chunks"
QUERY_SHARDS_DIR = "query_shards"

# Mining algorithm constants
TOPK_BUFFER_MULTIPLIER = 2  # Select 2x candidates to ensure enough negatives after filtering positives

# Default values for mining parameters
MINING_DEFAULTS = {
    "hard_negatives_to_mine": 20,
    "hard_neg_margin": 0.95,
    "hard_neg_margin_type": "perc",
    "mining_batch_size": 128,
    "query_embedding_batch_size": 16,
    "document_embedding_batch_size": 16,
    "corpus_chunk_size": 50000,
    "load_embeddings_from_cache": False,
    "use_negatives_from_file": False,
    # Prefix and length configuration for embedding generation
    "query_prefix": "",
    "passage_prefix": "",
    "query_max_length": 512,
    "passage_max_length": 512,
    # Tokenizer special-token behavior:
    #   - None (default): Use Automodel's tokenizer defaults (recommended for Automodel-trained models)
    #     This ensures mining behavior stays in sync with training as Automodel evolves.
    #   - True/False: Explicitly override for models trained in other frameworks.
    #     Use this ONLY when you know the external model used specific tokenizer settings.
    "add_bos_token": None,
    "add_eos_token": None,
    # Model loading parameters (loaded directly, not from config)
    "model_name_or_path": None,  # Required: path to model checkpoint
    "tokenizer_name_or_path": None,  # Optional: defaults to model_name_or_path
}


def build_distributed(cfg_dist: Dict[str, Any]) -> DistInfo:
    """Build and initialize distributed resources.

    Args:
        cfg_dist: Configuration for distributed environment.

    Returns:
        Distributed environment information from initialize_distributed.
    """
    backend = cfg_dist.get("backend", "nccl")
    timeout = cfg_dist.get("timeout_minutes", 1)
    return initialize_distributed(backend=backend, timeout_minutes=timeout)


def _load_npz_array(path: Path) -> np.ndarray:
    """Load numpy array from NPZ archive.

    Args:
        path: Path to NPZ file.

    Returns:
        Loaded numpy array.
    """
    cached = np.load(path)
    return cached[cached.files[0]]


def _compute_rank_partition(total_size: int, world_size: int, rank: int) -> Tuple[int, int]:
    """Compute contiguous partition boundaries for a given rank.

    Distributes `total_size` items across `world_size` ranks as evenly as possible,
    with remainder items distributed to lower ranks.

    Args:
        total_size: Total number of items to partition.
        world_size: Number of ranks.
        rank: Current rank (0-indexed).

    Returns:
        Tuple of (start_idx, end_idx) for this rank's partition.
    """
    base = total_size // world_size
    rem = total_size % world_size
    start_idx = rank * base + min(rank, rem)
    end_idx = start_idx + base + (1 if rank < rem else 0)
    return start_idx, end_idx


def _validate_shard_shape(shard_path: Path, expected_size: int, actual_size: int) -> None:
    """Validate that a shard has the expected number of items.

    Args:
        shard_path: Path to the shard file (for error reporting).
        expected_size: Expected number of items.
        actual_size: Actual number of items.

    Raises:
        ValueError: If sizes don't match.
    """
    if actual_size != expected_size:
        raise ValueError(f"Shard shape mismatch for {shard_path}: got {actual_size} items, expected {expected_size}")


class MineHardNegativesRecipe:
    """Recipe for mining hard negatives for biencoder training.

    This class orchestrates hard negative mining, from setup to mining execution.
    Hard negatives are documents that are semantically similar to the query but
    are not relevant, making them valuable for training more discriminative models.
    """

    def __init__(self, cfg):
        """Initialize the recipe with configuration.

        Args:
            cfg: Configuration object containing mining parameters.
                 The model is loaded directly from model_name_or_path,
                 not from cfg.model - this allows using saved checkpoints
                 without needing the full model architecture config.
        """
        self.cfg = cfg
        self.dist_env = None
        self.mining_cfg = None

        # Mining parameters (populated in setup)
        self.train_qa_file_path = None
        self.train_file_output_path = None
        self.cache_embeddings_dir = None
        self.hard_negatives_to_mine = None
        self.hard_neg_margin = None
        self.hard_neg_margin_type = None
        self.mining_batch_size = None
        self.query_embedding_batch_size = None
        self.document_embedding_batch_size = None
        self.corpus_chunk_size = None
        self.load_embeddings_from_cache = None
        self.use_negatives_from_file = None

        # Model loading parameters (populated in setup)
        self.model_name_or_path = None
        self.tokenizer_name_or_path = None
        self.add_bos_token = None
        self.add_eos_token = None

        # Model and tokenizer (populated in setup)
        self.model = None
        self.tokenizer = None

        # Data (populated in setup)
        self.questions_dataset = None
        self.documents_dataset = None
        self.corpus_path = None
        self.doc_to_idx = None
        self.idx_to_doc = None

        # Prepared data for mining (populated by _prepare_data)
        self.questions = None
        self.question_ids = None
        self.corpus_ids = None
        self.pos_doc_indices = None
        self.supplied_neg_doc_indices = None

        # Prefix and length configuration (populated in setup)
        self.query_prefix = None
        self.passage_prefix = None
        self.query_max_length = None
        self.passage_max_length = None

        # Embeddings (populated by _generate_embeddings)
        self.query_embeddings = None
        self.document_embeddings = None

        # Mining results (populated by _mine_hard_negatives)
        self.mined_neg_indices = None  # List[List[int]] - mined negative indices per query
        self.mined_neg_scores = None  # List[List[float]] - similarity scores for negatives
        self.pos_scores = None  # List[List[float]] - similarity scores for positives

    def setup(self):
        """Build all components needed for hard negative mining."""
        # Initialize distributed environment
        self.dist_env = build_distributed(self.cfg.get("dist_env", {}))

        # Validate and extract mining configuration
        self.mining_cfg = self.cfg.get("mining", None)
        if self.mining_cfg is None:
            raise ValueError(
                "Missing mining configuration. Provide mining parameters via command line:\n"
                "  --mining.train_qa_file_path /path/to/input.json\n"
                "  --mining.train_file_output_path /path/to/output.json\n"
                "  --mining.cache_embeddings_dir /path/to/cache (optional)\n"
                "  --mining.hard_neg_margin 0.95 (optional, default: 0.95)"
            )

        # Extract mining parameters with defaults
        self._extract_mining_params()

        # Validate required parameters
        self._validate_mining_params()

        # Load model directly from checkpoint path
        # This loads the saved model without requiring architecture config
        logger.info(f"Loading biencoder model from {self.model_name_or_path}...")
        self.model = NeMoAutoModelBiencoder.from_pretrained(
            self.model_name_or_path,
            # Use inference-appropriate settings
            use_liger_kernel=False,  # Not needed for inference
            use_sdpa_patching=True,
        )
        self.model = self.model.to(self.dist_env.device)
        self.model.eval()

        # Load and configure tokenizer
        self._configure_tokenizer()

        # Load dataset and corpus
        self._load_data()

        # Build document-to-index mappings
        self._build_document_mappings()

        # Prepare data for mining (extract queries, IDs, indices)
        self._prepare_data()

    def _get_mining_param(self, name, default=None):
        """Get mining parameter from config, with fallback to defaults.

        Args:
            name: Parameter name.
            default: Default value if not in config or MINING_DEFAULTS.

        Returns:
            Parameter value.
        """
        if default is None:
            default = MINING_DEFAULTS.get(name)
        return self.mining_cfg.get(name, default)

    def _extract_mining_params(self):
        """Extract all mining parameters from configuration."""
        # Required parameters
        self.train_qa_file_path = self._get_mining_param("train_qa_file_path")
        self.train_file_output_path = self._get_mining_param("train_file_output_path")
        self.model_name_or_path = self._get_mining_param("model_name_or_path")

        # Optional parameters with defaults
        self.cache_embeddings_dir = self._get_mining_param("cache_embeddings_dir")
        self.hard_negatives_to_mine = self._get_mining_param("hard_negatives_to_mine")
        self.hard_neg_margin = self._get_mining_param("hard_neg_margin")
        self.hard_neg_margin_type = self._get_mining_param("hard_neg_margin_type")
        self.mining_batch_size = self._get_mining_param("mining_batch_size")
        self.query_embedding_batch_size = self._get_mining_param("query_embedding_batch_size")
        self.document_embedding_batch_size = self._get_mining_param("document_embedding_batch_size")
        self.corpus_chunk_size = self._get_mining_param("corpus_chunk_size")
        self.load_embeddings_from_cache = self._get_mining_param("load_embeddings_from_cache")
        self.use_negatives_from_file = self._get_mining_param("use_negatives_from_file")

        # Model loading: tokenizer defaults to model path if not specified
        self.tokenizer_name_or_path = self._get_mining_param("tokenizer_name_or_path")
        if self.tokenizer_name_or_path is None:
            self.tokenizer_name_or_path = self.model_name_or_path

        # Tokenizer special token behavior (optional; defaults to tokenizer behavior if None)
        self.add_bos_token = self._get_mining_param("add_bos_token")
        self.add_eos_token = self._get_mining_param("add_eos_token")

        # Prefix and length parameters for embedding generation
        self.query_prefix = self._get_mining_param("query_prefix")
        self.passage_prefix = self._get_mining_param("passage_prefix")
        self.query_max_length = self._get_mining_param("query_max_length")
        self.passage_max_length = self._get_mining_param("passage_max_length")

    def _validate_mining_params(self):
        """Validate required mining parameters.

        Raises:
            ValueError: If any required parameter is missing or invalid.
        """
        if self.train_qa_file_path is None:
            raise ValueError("Missing required parameter: --mining.train_qa_file_path")
        if self.train_file_output_path is None:
            raise ValueError("Missing required parameter: --mining.train_file_output_path")
        if self.model_name_or_path is None:
            raise ValueError("Missing required parameter: --mining.model_name_or_path")

        # Validate margin type if margin is specified
        if self.hard_neg_margin is not None:
            valid_types = ["perc", "abs"]
            if self.hard_neg_margin_type.lower() not in valid_types:
                raise ValueError(
                    f"Invalid hard_neg_margin_type: {self.hard_neg_margin_type}. Must be one of {valid_types}"
                )

    def _configure_tokenizer(self):
        """Load and configure tokenizer with appropriate settings."""
        logger.info(f"Loading tokenizer from {self.tokenizer_name_or_path}...")

        # Build tokenizer kwargs
        tokenizer_kwargs = {}
        if self.add_bos_token is not None:
            tokenizer_kwargs["add_bos_token"] = self.add_bos_token
        if self.add_eos_token is not None:
            tokenizer_kwargs["add_eos_token"] = self.add_eos_token

        # Load tokenizer
        self.tokenizer = NeMoAutoTokenizer.from_pretrained(self.tokenizer_name_or_path, **tokenizer_kwargs)

        # Log tokenizer configuration for transparency
        actual_bos = getattr(self.tokenizer, "add_bos_token", None)
        actual_eos = getattr(self.tokenizer, "add_eos_token", None)

        if self.add_bos_token is None and self.add_eos_token is None:
            logger.info(f"Using Automodel tokenizer defaults: add_bos_token={actual_bos}, add_eos_token={actual_eos}")
        else:
            logger.info(
                f"Using explicit tokenizer settings: "
                f"add_bos_token={self.add_bos_token}, add_eos_token={self.add_eos_token} "
                f"(overriding Automodel defaults)"
            )

        # Set pad token if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"

    def _synchronize_ranks(self):
        """Synchronize all distributed ranks with a barrier.

        Handles device-specific barrier calls for CUDA vs CPU.
        """
        if torch.distributed.is_initialized():
            if self.dist_env.device.type == "cuda" and self.dist_env.device.index is not None:
                torch.distributed.barrier(device_ids=[self.dist_env.device.index])
            else:
                torch.distributed.barrier()

    def _load_data(self):
        """Load dataset and corpus from the input QA file.

        Uses load_datasets() from retrieval_dataset.py to load the questions
        dataset and corpus dictionary. Validates that only a single corpus
        is referenced.
        """
        logger.info(f"Loading dataset from {self.train_qa_file_path}")
        dataset, corpus_dict = load_datasets(self.train_qa_file_path)

        # Verify corpus points to a single path
        if len(corpus_dict) != 1:
            raise ValueError(
                f"Mining requires exactly one corpus, but found {len(corpus_dict)} corpora. "
                f"Corpus paths: {list(corpus_dict.keys())}"
            )

        self.corpus_path = list(corpus_dict.keys())[0]
        self.documents_dataset = corpus_dict[self.corpus_path]
        self.questions_dataset = dataset

        logger.info(f"Loaded {len(dataset)} questions from corpus: {self.corpus_path}")

    def _build_document_mappings(self):
        """Build bidirectional mappings between document IDs and indices.

        Creates doc_to_idx and idx_to_doc dictionaries for efficient lookup
        during the mining process. Documents are sorted by ID for deterministic
        ordering across runs.
        """
        all_doc_ids = sorted(self.documents_dataset.get_all_ids())
        self.doc_to_idx = {doc_id: i for i, doc_id in enumerate(all_doc_ids)}
        self.idx_to_doc = {i: doc_id for i, doc_id in enumerate(all_doc_ids)}

        logger.info(f"Built mappings for {len(all_doc_ids)} documents")

    def _prepare_data(self):
        """Extract query texts and document indices from questions dataset.

        Iterates through the questions dataset and extracts:
        - Query texts (with EMPTY_QUESTION placeholder for empty queries)
        - Question IDs
        - Corpus IDs
        - Positive document indices (mapped via doc_to_idx)
        - Supplied negative document indices (if use_negatives_from_file is True)
        """
        questions = []
        question_ids = []
        corpus_ids = []
        positive_document_indices = []
        supplied_negative_document_indices = []

        for row in tqdm(self.questions_dataset, desc="Processing questions"):
            # Handle empty questions with placeholder
            question_text = row["question"]
            if not question_text:
                question_text = EMPTY_QUESTION

            questions.append(question_text)
            question_ids.append(row["question_id"])
            corpus_ids.append(row["corpus_id"])

            # Map positive doc IDs to indices
            pos_indices = [self.doc_to_idx[doc["id"]] for doc in row["pos_doc"]]
            positive_document_indices.append(pos_indices)

            # Map negative doc IDs to indices (only if use_negatives_from_file)
            neg_indices = []
            if self.use_negatives_from_file and row.get("neg_doc"):
                neg_indices = [self.doc_to_idx[doc["id"]] for doc in row["neg_doc"]]
            supplied_negative_document_indices.append(neg_indices)

        assert len(questions) == len(question_ids)

        self.questions = questions
        self.question_ids = question_ids
        self.corpus_ids = corpus_ids
        self.pos_doc_indices = positive_document_indices
        self.supplied_neg_doc_indices = supplied_negative_document_indices

        logger.info(f"Prepared {len(questions)} questions for mining")

    # =========================================================================
    # Embedding Generation Methods
    # =========================================================================

    def _get_document_text(self, doc: dict) -> str:
        """Extract text from document dict, checking common field names.

        Args:
            doc: Document dictionary from corpus.

        Returns:
            Document text string.
        """
        # If a title is present, concatenate "{title} {text}", otherwise use "{text}".
        # Always strip whitespace.
        if "title" in doc:
            return (str(doc["title"]) + " " + doc["text"]).strip()
        return doc["text"].strip()

    def _encode_texts(
        self,
        texts: List[str],
        batch_size: int,
        max_length: int,
        prefix: str = "",
        encoder_type: str = "query",
    ) -> np.ndarray:
        """Encode texts into embeddings.

        Args:
            texts: List of text strings to encode.
            batch_size: Batch size for encoding.
            max_length: Maximum sequence length for tokenization.
            prefix: Optional prefix to prepend to each text.
            encoder_type: "query" to use lm_q, "passage" to use lm_p.

        Returns:
            numpy array of embeddings [num_texts, embedding_dim].
        """
        embeddings = []
        num_texts = len(texts)
        num_batches = (num_texts + batch_size - 1) // batch_size

        for i in tqdm(range(0, num_texts, batch_size), desc=f"Encoding {encoder_type}", total=num_batches):
            batch_texts = texts[i : i + batch_size]

            # Apply prefix if provided
            if prefix:
                batch_texts = [prefix + text for text in batch_texts]

            # Tokenize without return_tensors first (NeMo tokenizer compatibility)
            tokenized = self.tokenizer(
                batch_texts,
                max_length=max_length,
                truncation=True,
                padding=False,
                return_token_type_ids=False,
            )

            # Convert to list of dicts for padding
            tokenized_list = [{k: tokenized[k][j] for k in tokenized.keys()} for j in range(len(batch_texts))]

            # Pad and convert to tensors
            inputs = self.tokenizer.pad(
                tokenized_list,
                padding="longest",
                return_tensors="pt",
            )
            inputs = {k: v.to(self.dist_env.device) for k, v in inputs.items()}

            # Encode using the appropriate encoder
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                batch_embeds = self.model.encode(inputs, encoder=encoder_type)

            embeddings.append(batch_embeds.cpu().float().numpy())

        return np.concatenate(embeddings, axis=0)

    def _encode_queries(self) -> np.ndarray:
        """Encode all queries into embeddings.

        Uses self.query_prefix and self.query_max_length from mining config.

        Returns:
            numpy array of query embeddings [num_queries, embedding_dim].
        """
        return self._encode_texts(
            texts=self.questions,
            batch_size=self.query_embedding_batch_size,
            max_length=self.query_max_length,
            prefix=self.query_prefix,
            encoder_type="query",
        )

    def _encode_queries_sharded(self) -> np.ndarray:
        """Encode queries sharded across ranks and assemble on rank0.

        This is useful when the number of queries is large (e.g., 100k+), and we want
        to utilize multiple GPUs for query embedding generation without sharding mining/scoring.

        Requires cache_embeddings_dir so ranks can write shard files and rank0 can assemble.
        """
        if self.dist_env.world_size == 1:
            return self._encode_queries()

        if not self.cache_embeddings_dir:
            raise ValueError(
                "Distributed query encoding requires --mining.cache_embeddings_dir so ranks can write query shards "
                "and rank0 can assemble them."
            )

        cache_dir = Path(self.cache_embeddings_dir)
        shard_dir = cache_dir / QUERY_SHARDS_DIR
        shard_dir.mkdir(parents=True, exist_ok=True)

        num_q = len(self.questions)
        ws = self.dist_env.world_size
        r = self.dist_env.rank

        # Compute this rank's partition
        local_start, local_end = _compute_rank_partition(num_q, ws, r)
        shard_path = shard_dir / f"queries_rank{r:04d}.npz"

        # Compute or load this rank's shard
        if shard_path.exists():
            local_embeds = _load_npz_array(shard_path)
        else:
            local_texts = self.questions[local_start:local_end]
            local_embeds = self._encode_texts(
                texts=local_texts,
                batch_size=self.query_embedding_batch_size,
                max_length=self.query_max_length,
                prefix=self.query_prefix,
                encoder_type="query",
            )
            np.savez(shard_path, local_embeds)

        # Synchronize so rank0 can safely assemble
        self._synchronize_ranks()

        if not self.dist_env.is_main:
            return np.empty((0, 0), dtype=np.float32)

        # Assemble in rank order to preserve the original query order.
        parts: List[np.ndarray] = []
        for rr in range(ws):
            rr_path = shard_dir / f"queries_rank{rr:04d}.npz"
            if not rr_path.exists():
                raise FileNotFoundError(f"Missing query shard cache: {rr_path}")

            rr_emb = _load_npz_array(rr_path)
            rr_start, rr_end = _compute_rank_partition(num_q, ws, rr)
            expected = rr_end - rr_start
            _validate_shard_shape(rr_path, expected, rr_emb.shape[0])
            parts.append(rr_emb)

        return np.concatenate(parts, axis=0)

    def _load_cached_chunk(self, cache_path: Path) -> Optional[np.ndarray]:
        """Load a fully-assembled chunk cache if it exists.

        In distributed mode, only rank0 loads the cache to avoid redundant IO.

        Args:
            cache_path: Path to cached chunk file.

        Returns:
            Cached embeddings array, or None if cache doesn't exist.
        """
        if cache_path is None or not cache_path.exists():
            return None

        # In distributed runs, only rank0 needs the assembled chunk
        if self.dist_env.world_size > 1 and not self.dist_env.is_main:
            return np.empty((0, 0), dtype=np.float32)

        return _load_npz_array(cache_path)

    def _encode_chunk_distributed(
        self,
        texts: List[str],
        cache_path: Path,
    ) -> np.ndarray:
        """Encode a chunk of documents in distributed mode.

        Shards the documents within the chunk across ranks, encodes each shard,
        and assembles on rank0.

        Args:
            texts: Document texts to encode.
            cache_path: Path for caching the assembled chunk.

        Returns:
            Assembled document embeddings for this chunk (rank0 only).
        """
        num_docs_in_chunk = len(texts)
        ws = self.dist_env.world_size
        r = self.dist_env.rank

        # Compute this rank's partition
        local_start, local_end = _compute_rank_partition(num_docs_in_chunk, ws, r)

        # Per-rank cache for this chunk
        rank_cache_path = cache_path.parent / f"{cache_path.stem}_rank{r:04d}{cache_path.suffix}"

        # Compute or load this rank's slice
        if rank_cache_path.exists():
            local_embeds = _load_npz_array(rank_cache_path)
        else:
            local_texts = texts[local_start:local_end]
            local_embeds = self._encode_texts(
                texts=local_texts,
                batch_size=self.document_embedding_batch_size,
                max_length=self.passage_max_length,
                prefix=self.passage_prefix,
                encoder_type="passage",
            )
            np.savez(rank_cache_path, local_embeds)

        # Synchronize to ensure all rank shard files exist before assembly on rank0
        self._synchronize_ranks()

        # Assemble full chunk on rank0 (in the original doc_indices order)
        if self.dist_env.is_main:
            parts: List[np.ndarray] = []
            for rr in range(ws):
                rr_start, rr_end = _compute_rank_partition(num_docs_in_chunk, ws, rr)
                rr_path = cache_path.parent / f"{cache_path.stem}_rank{rr:04d}{cache_path.suffix}"
                if not rr_path.exists():
                    raise FileNotFoundError(f"Missing rank shard cache for chunk: {rr_path}")

                rr_emb = _load_npz_array(rr_path)
                expected = rr_end - rr_start
                _validate_shard_shape(rr_path, expected, rr_emb.shape[0])
                parts.append(rr_emb)

            embeddings = np.concatenate(parts, axis=0)
            # Save assembled chunk for faster reuse next time
            np.savez(cache_path, embeddings)
            return embeddings

        # Non-main ranks do not need the assembled chunk
        return np.empty((0, 0), dtype=np.float32)

    def _encode_chunk_local(
        self,
        texts: List[str],
        cache_path: Optional[Path],
    ) -> np.ndarray:
        """Encode a chunk of documents locally (single-process).

        Args:
            texts: Document texts to encode.
            cache_path: Optional path for caching.

        Returns:
            Document embeddings for this chunk.
        """
        embeddings = self._encode_texts(
            texts=texts,
            batch_size=self.document_embedding_batch_size,
            max_length=self.passage_max_length,
            prefix=self.passage_prefix,
            encoder_type="passage",
        )
        if cache_path is not None:
            np.savez(cache_path, embeddings)
        return embeddings

    def _encode_documents_chunk(
        self,
        doc_indices: List[int],
        cache_path: Optional[Path] = None,
    ) -> np.ndarray:
        """Encode a chunk of documents into embeddings.

        Args:
            doc_indices: List of document indices to encode.
            cache_path: Optional path to save/load chunk cache.

        Returns:
            numpy array of document embeddings [num_docs, embedding_dim].
        """
        # Fast path: load from cache if available
        cached_result = self._load_cached_chunk(cache_path)
        if cached_result is not None:
            return cached_result

        # Fetch document texts
        doc_ids = [self.idx_to_doc[idx] for idx in doc_indices]
        docs = [self.documents_dataset.get_document_by_id(doc_id) for doc_id in doc_ids]
        texts = [self._get_document_text(doc) for doc in docs]

        # Encode: distributed or local
        if self.dist_env.world_size > 1:
            if cache_path is None:
                raise ValueError(
                    "Distributed mining requires --mining.cache_embeddings_dir so ranks can shard document encoding "
                    "and rank0 can assemble embeddings from cached corpus chunks."
                )
            return self._encode_chunk_distributed(texts, cache_path)
        else:
            return self._encode_chunk_local(texts, cache_path)

    def _encode_all_documents(self) -> np.ndarray:
        """Encode all documents in corpus, chunk by chunk.

        Returns:
            numpy array of document embeddings [num_docs, embedding_dim].
        """
        # Setup cache directory if caching enabled
        chunk_cache_dir = None
        if self.cache_embeddings_dir:
            chunk_cache_dir = Path(self.cache_embeddings_dir) / CORPUS_CHUNKS_DIR
            chunk_cache_dir.mkdir(parents=True, exist_ok=True)
        elif self.dist_env.world_size > 1:
            raise ValueError(
                "Distributed mining requires --mining.cache_embeddings_dir so ranks can shard document encoding "
                "and rank0 can assemble embeddings from cached corpus chunks."
            )

        total_docs = len(self.idx_to_doc)
        all_embeddings = []  # only populated on rank0

        # Process documents in chunks
        chunk_idx = 0
        num_chunks = (total_docs + self.corpus_chunk_size - 1) // self.corpus_chunk_size
        for start in tqdm(
            range(0, total_docs, self.corpus_chunk_size),
            desc="Encoding document chunks",
            total=num_chunks,
            disable=not self.dist_env.is_main,
        ):
            end = min(start + self.corpus_chunk_size, total_docs)
            doc_indices = list(range(start, end))

            cache_path = None
            if chunk_cache_dir:
                cache_path = chunk_cache_dir / f"chunk_{chunk_idx:04d}.npz"

            chunk_embeddings = self._encode_documents_chunk(doc_indices, cache_path)
            if self.dist_env.is_main:
                all_embeddings.append(chunk_embeddings)

            chunk_idx += 1

        if self.dist_env.is_main:
            return np.concatenate(all_embeddings, axis=0)
        return np.empty((0, 0), dtype=np.float32)

    def _load_embeddings_from_cache(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Load query and document embeddings from cache.

        Returns:
            Tuple of (query_embeddings, document_embeddings), or (None, None) if not found.
        """
        if not self.cache_embeddings_dir:
            return None, None

        cache_dir = Path(self.cache_embeddings_dir)
        query_path = cache_dir / QUERY_EMBEDDINGS_FNAME
        doc_path = cache_dir / DOCUMENT_EMBEDDINGS_FNAME

        if not query_path.exists() or not doc_path.exists():
            return None, None

        query_embeddings = _load_npz_array(query_path)
        doc_embeddings = _load_npz_array(doc_path)

        return query_embeddings, doc_embeddings

    def _has_full_embeddings_cache(self) -> bool:
        """Check if the consolidated (rank0) embedding cache exists.

        This is intentionally a lightweight existence check (no file reads), used to avoid
        redundant IO on non-main ranks in distributed runs.
        """
        if not self.cache_embeddings_dir:
            return False
        cache_dir = Path(self.cache_embeddings_dir)
        return (cache_dir / QUERY_EMBEDDINGS_FNAME).exists() and (cache_dir / DOCUMENT_EMBEDDINGS_FNAME).exists()

    def _save_embeddings_to_cache(
        self,
        query_embeddings: np.ndarray,
        document_embeddings: np.ndarray,
    ) -> None:
        """Save query and document embeddings to cache.

        Args:
            query_embeddings: Query embeddings array.
            document_embeddings: Document embeddings array.
        """
        if not self.cache_embeddings_dir:
            return

        cache_dir = Path(self.cache_embeddings_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        np.savez(cache_dir / QUERY_EMBEDDINGS_FNAME, query_embeddings)
        np.savez(cache_dir / DOCUMENT_EMBEDDINGS_FNAME, document_embeddings)

        logger.info(f"Saved embeddings to cache: {cache_dir}")

    def _generate_embeddings(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate embeddings for queries and documents.

        Handles caching and orchestrates the encoding process.

        Returns:
            Tuple of (query_embeddings, document_embeddings).
        """
        # Try loading from cache first.
        #
        # In distributed runs, only rank0 needs the consolidated embeddings for mining.
        # To avoid redundant IO, rank0 checks for cache presence and broadcasts a cache_hit flag;
        # only rank0 reads the cache files.
        if self.load_embeddings_from_cache and self.cache_embeddings_dir:
            cache_hit = False
            if self.dist_env.world_size > 1 and torch.distributed.is_initialized():
                if self.dist_env.is_main:
                    cache_hit = self._has_full_embeddings_cache()
                # Broadcast decision from rank0 to all ranks (NCCL requires CUDA tensors).
                flag_device = self.dist_env.device if self.dist_env.device.type == "cuda" else torch.device("cpu")
                flag = torch.tensor([1 if cache_hit else 0], dtype=torch.int64, device=flag_device)
                torch.distributed.broadcast(flag, src=0)
                cache_hit = bool(flag.item())
            else:
                cache_hit = self._has_full_embeddings_cache()

            if cache_hit:
                if self.dist_env.is_main:
                    logger.info("Loading embeddings from cache (rank0 only)...")
                    query_embeddings, document_embeddings = self._load_embeddings_from_cache()
                    assert query_embeddings is not None and document_embeddings is not None
                    logger.info(
                        f"Loaded embeddings from cache: queries={query_embeddings.shape}, "
                        f"documents={document_embeddings.shape}"
                    )
                    return query_embeddings, document_embeddings
                # Non-main ranks do not need to read large cache files.
                return np.empty((0, 0), dtype=np.float32), np.empty((0, 0), dtype=np.float32)

            if self.dist_env.is_main:
                logger.info("Cache not found or incomplete, generating embeddings...")

        # Generate query embeddings (shard across ranks if distributed)
        query_embeddings = None
        logger.info(f"Encoding {len(self.questions)} queries...")
        if self.dist_env.world_size > 1:
            query_embeddings = self._encode_queries_sharded()
        else:
            query_embeddings = self._encode_queries()
        if self.dist_env.is_main:
            logger.info(f"Query embeddings shape: {query_embeddings.shape}")

        # Generate document embeddings
        num_docs = len(self.idx_to_doc)
        logger.info(f"Encoding {num_docs} documents...")
        document_embeddings = self._encode_all_documents()
        if self.dist_env.is_main:
            logger.info(f"Document embeddings shape: {document_embeddings.shape}")

        # Save to cache (rank0 only; writes the consolidated query_embeddings.npz and passage_embeddings.npz)
        if self.cache_embeddings_dir and self.dist_env.is_main:
            self._save_embeddings_to_cache(query_embeddings, document_embeddings)

        # Only rank0 proceeds to mining/output; other ranks return dummy arrays.
        if not self.dist_env.is_main:
            return np.empty((0, 0), dtype=np.float32), np.empty((0, 0), dtype=np.float32)

        assert query_embeddings is not None
        return query_embeddings, document_embeddings

    def _unload_model(self):
        """Unload the model from GPU memory after embedding generation.

        This frees up GPU memory for the mining phase, which only operates
        on embeddings and doesn't need the model parameters.

        Model metadata (pooling, l2_normalize) is extracted before unloading
        to ensure it's available for output generation.
        """
        if self.model is None:
            return

        # Extract metadata needed for output before unloading
        if not hasattr(self, "_model_pooling"):
            self._model_pooling = self.model.pooling
            self._model_l2_normalize = self.model.l2_normalize

        logger.info("Unloading model to free GPU memory for mining...")

        # Move model to CPU first (safer cleanup)
        self.model = self.model.cpu()

        # Delete model reference
        del self.model
        self.model = None

        # Clear CUDA cache
        torch.cuda.empty_cache()

        if self.dist_env.is_main:
            allocated = torch.cuda.memory_allocated(self.dist_env.device) / 1e9
            logger.info(f"Model unloaded. GPU memory allocated: {allocated:.2f}GB")

    # =========================================================================
    # Hard Negative Mining Methods
    # =========================================================================

    def _mine_hard_negatives(
        self,
        query_embeddings: np.ndarray,
        document_embeddings: np.ndarray,
        pos_doc_indices: List[List[int]],
        batch_size: int,
        num_negs: int,
        hard_neg_margin: Optional[float] = None,
        hard_neg_margin_type: Optional[str] = None,
    ) -> Tuple[List[List[int]], List[List[float]], List[List[float]]]:
        """Mine hard negatives for each query.

        This implementation uses the following key behaviors:
        - Deduplicates positive indices before masking (avoids double-masking)
        - Preserves original order of positive scores (matches pos_doc order in input)
        - Uses vectorized batch-level margin filtering for efficiency
        - Uses batch-level topk for efficiency

        Args:
            query_embeddings: Query embeddings [num_queries, embedding_dim].
            document_embeddings: Document embeddings [num_docs, embedding_dim].
            pos_doc_indices: List of positive document indices for each query.
            batch_size: Number of queries to process per batch.
            num_negs: Number of hard negatives to mine per query.
            hard_neg_margin: Optional margin for filtering false negatives.
            hard_neg_margin_type: "perc" (percentage) or "abs" (absolute).

        Returns:
            Tuple of:
                - neg_indices: List of hard negative indices per query
                - neg_scores: Similarity scores for each hard negative
                - pos_scores: Similarity scores for each positive document
        """
        # Convert document embeddings to tensor once (biencoder embeddings are 2D)
        doc_embeddings_tensor = torch.tensor(document_embeddings, device="cuda")

        neg_indices_all = []
        neg_scores_all = []
        pos_scores_all = []

        num_queries = query_embeddings.shape[0]
        num_batches = (num_queries + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(num_batches), desc="Mining hard negatives"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_queries)

            # Get batch of query embeddings and positive indices
            batch_query_embs = query_embeddings[start_idx:end_idx]
            batch_pos_indices = pos_doc_indices[start_idx:end_idx]

            # Compute similarity scores: [batch_size, num_docs]
            batch_query_tensor = torch.tensor(batch_query_embs, device="cuda")
            batch_scores = batch_query_tensor @ doc_embeddings_tensor.T

            # Extract positive scores and mask positives
            min_pos_scores = []  # Minimum positive score per query (for margin filtering)
            batch_pos_scores = []

            for i, pos_indices in enumerate(batch_pos_indices):
                # Use dict to store scores by index (for deduplication and order preservation)
                scores_by_idx = {}

                # Deduplicate positive indices before processing
                for pos_idx in list(set(pos_indices)):
                    score = batch_scores[i, pos_idx].item()
                    scores_by_idx[pos_idx] = score
                    # Mask out positive documents
                    batch_scores[i, pos_idx] = float("-inf")

                # Reconstruct scores in original order (preserves pos_doc order for output)
                query_pos_scores = [scores_by_idx[pos_idx] for pos_idx in pos_indices]
                batch_pos_scores.append(query_pos_scores)

                # Track minimum positive score for margin filtering
                if query_pos_scores:
                    min_pos_scores.append(min(scores_by_idx.values()))
                else:
                    min_pos_scores.append(0.0)  # Handle edge case of no positives

            # Vectorized margin filtering
            if hard_neg_margin is not None:
                min_pos_tensor = torch.tensor(min_pos_scores, device="cuda")

                if hard_neg_margin_type.lower() == "abs":
                    threshold = torch.unsqueeze(min_pos_tensor - hard_neg_margin, dim=1)
                elif hard_neg_margin_type.lower() == "perc":
                    threshold = torch.unsqueeze(min_pos_tensor * hard_neg_margin, dim=1)
                else:
                    threshold = None

                if threshold is not None:
                    downscore_mask = batch_scores > threshold
                    batch_scores[downscore_mask] = float("-inf")

            # Batch-level top-k selection
            k = min(num_negs * TOPK_BUFFER_MULTIPLIER, batch_scores.shape[1])
            topk = batch_scores.topk(k=k, dim=1)
            topk_indices = topk.indices.tolist()

            # Post-process: remove any remaining positives and limit to num_negs
            for i, query_pos_indices in enumerate(batch_pos_indices):
                pos_set = set(query_pos_indices)

                # Filter out positives from top-k candidates
                hard_neg_candidates = [idx for idx in topk_indices[i] if idx not in pos_set]
                hard_neg_scores = [batch_scores[i, idx].item() for idx in topk_indices[i] if idx not in pos_set]

                # Limit to num_negs
                neg_indices_all.append(hard_neg_candidates[:num_negs])
                neg_scores_all.append(hard_neg_scores[:num_negs])

            pos_scores_all.extend(batch_pos_scores)

        return neg_indices_all, neg_scores_all, pos_scores_all

    # =========================================================================
    # Output Generation Methods
    # =========================================================================

    def _get_mining_args_dict(self) -> Dict[str, Any]:
        """Get dictionary of mining arguments for output metadata.

        Returns:
            Dict containing all mining parameters for reproducibility.
        """
        return {
            "train_qa_file_path": str(self.train_qa_file_path),
            "train_file_output_path": str(self.train_file_output_path),
            "cache_embeddings_dir": str(self.cache_embeddings_dir) if self.cache_embeddings_dir else None,
            "hard_negatives_to_mine": self.hard_negatives_to_mine,
            "hard_neg_margin": self.hard_neg_margin,
            "hard_neg_margin_type": self.hard_neg_margin_type,
            "mining_batch_size": self.mining_batch_size,
            "query_embedding_batch_size": self.query_embedding_batch_size,
            "document_embedding_batch_size": self.document_embedding_batch_size,
            "corpus_chunk_size": self.corpus_chunk_size,
            "use_negatives_from_file": self.use_negatives_from_file,
            "query_prefix": self.query_prefix,
            "passage_prefix": self.passage_prefix,
            "query_max_length": self.query_max_length,
            "passage_max_length": self.passage_max_length,
            "add_bos_token": self.add_bos_token,  # None means "use Automodel tokenizer defaults"
            "add_eos_token": self.add_eos_token,  # None means "use Automodel tokenizer defaults"
            # Model info (loaded directly from path, not from config)
            "model_name_or_path": str(self.model_name_or_path),
            "tokenizer_name_or_path": str(self.tokenizer_name_or_path),
            "pooling": self._model_pooling if hasattr(self, "_model_pooling") else self.model.pooling,
            "l2_normalize": self._model_l2_normalize
            if hasattr(self, "_model_l2_normalize")
            else self.model.l2_normalize,
        }

    def _build_negative_docs_by_question_id(self) -> Dict[str, List[Dict[str, Any]]]:
        """Build mapping from question_id to mined negative documents with scores.

        If use_negatives_from_file is True, supplied negatives are prepended
        to mined negatives (with score=-1 since we don't compute their scores).

        Returns:
            Dict mapping question_id to list of {"id": doc_id, "score": score} dicts.
        """
        negative_docs_by_question_id = {}

        for i, question_id in enumerate(self.question_ids):
            neg_docs = []

            # Include supplied negatives first (if enabled)
            if self.use_negatives_from_file and self.supplied_neg_doc_indices[i]:
                for neg_idx in self.supplied_neg_doc_indices[i]:
                    neg_docs.append(
                        {
                            "id": self.idx_to_doc[neg_idx],
                            "score": -1,  # Score unknown for supplied negatives
                        }
                    )

            # Add mined negatives with scores
            for neg_idx, score in zip(self.mined_neg_indices[i], self.mined_neg_scores[i]):
                neg_docs.append({"id": self.idx_to_doc[neg_idx], "score": score})

            negative_docs_by_question_id[question_id] = neg_docs

        return negative_docs_by_question_id

    def _build_positive_scores_by_question_id(self) -> Dict[str, List[float]]:
        """Build mapping from question_id to positive document scores.

        Scores are in the same order as pos_doc in the original data.

        Returns:
            Dict mapping question_id to list of positive scores.
        """
        return {question_id: scores for question_id, scores in zip(self.question_ids, self.pos_scores)}

    def _write_output(self) -> None:
        """Write the output JSON file with mined hard negatives.

        The output format:
        - Preserves all top-level keys from input (corpus, etc.)
        - Adds mining metadata section with parameters used
        - Replaces neg_doc with newly mined negatives
        - Adds similarity scores to all documents (pos_doc and neg_doc)
        - Removes legacy score fields (pos_score, neg_scores) if present
        """
        import json

        # Load original input file (preserves all top-level keys like corpus)
        with open(self.train_qa_file_path, "r") as f:
            output = json.load(f)

        # Build lookup dictionaries
        neg_docs_by_qid = self._build_negative_docs_by_question_id()
        pos_scores_by_qid = self._build_positive_scores_by_question_id()

        # Add mining metadata
        output["mining"] = {"args": self._get_mining_args_dict()}

        # Clear data and rebuild with enriched rows
        output["data"] = []

        # Iterate through original dataset and enrich with mining results
        for row in self.questions_dataset:
            question_id = row["question_id"]

            # Replace neg_doc with mined negatives
            row["neg_doc"] = neg_docs_by_qid.get(question_id, [])

            # Add scores to positive docs
            pos_scores = pos_scores_by_qid.get(question_id, [])
            for j, pos_doc in enumerate(row.get("pos_doc", [])):
                if j < len(pos_scores):
                    pos_doc["score"] = pos_scores[j]

            # Remove legacy score fields if present
            if "pos_score" in row:
                row.pop("pos_score")
            if "neg_scores" in row:
                row.pop("neg_scores")

            output["data"].append(row)

        # Ensure output directory exists
        output_path = Path(self.train_file_output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write output file with formatting
        with open(output_path, "w") as f:
            json.dump(output, f, indent=4, ensure_ascii=False)

        logger.info(f"Output written to {output_path}")

    # =========================================================================
    # Run and Configuration Methods
    # =========================================================================

    def run(self):
        """Run the hard negative mining pipeline.

        Generates query and document embeddings, mines hard negatives using
        similarity scores with margin filtering, and writes the output file.
        """
        if self.dist_env.is_main:
            self._print_configuration()

        # Generate embeddings
        logger.info("Generating embeddings...")
        self.query_embeddings, self.document_embeddings = self._generate_embeddings()

        if self.dist_env.is_main:
            logger.info(f"Query embeddings: {self.query_embeddings.shape}")
            logger.info(f"Document embeddings: {self.document_embeddings.shape}")

        # Unload model after embeddings are generated to free GPU memory
        self._unload_model()

        # Synchronize all ranks after model cleanup (if distributed)
        self._synchronize_ranks()

        # Mine hard negatives (only on main process)
        if self.dist_env.is_main:
            logger.info("Mining hard negatives...")
            self.mined_neg_indices, self.mined_neg_scores, self.pos_scores = self._mine_hard_negatives(
                query_embeddings=self.query_embeddings,
                document_embeddings=self.document_embeddings,
                pos_doc_indices=self.pos_doc_indices,
                batch_size=self.mining_batch_size,
                num_negs=self.hard_negatives_to_mine,
                hard_neg_margin=self.hard_neg_margin,
                hard_neg_margin_type=self.hard_neg_margin_type,
            )

            # Log mining statistics
            total_mined = sum(len(negs) for negs in self.mined_neg_indices)
            avg_mined = total_mined / len(self.mined_neg_indices) if self.mined_neg_indices else 0

            logger.info(f"Mined {total_mined} hard negatives for {len(self.mined_neg_indices)} queries")
            logger.info(f"Average negatives per query: {avg_mined:.1f}")

            # Write output
            logger.info("Writing output...")
            self._write_output()

            # Print completion summary
            print("\n" + "=" * 60)
            print("Hard Negative Mining Complete")
            print("=" * 60)
            print(f"  Total queries:           {len(self.mined_neg_indices)}")
            print(f"  Total hard negatives:    {total_mined}")
            print(f"  Avg negatives per query: {avg_mined:.1f}")
            print(f"  Output file:             {self.train_file_output_path}")
            print("=" * 60)

    def _print_configuration(self):
        """Print mining configuration summary."""
        print("=" * 60)
        print("Hard Negative Mining - Configuration")
        print("=" * 60)
        print("\nMining configuration:")
        print(f"  train_qa_file_path:            {self.train_qa_file_path}")
        print(f"  train_file_output_path:        {self.train_file_output_path}")
        print(f"  cache_embeddings_dir:          {self.cache_embeddings_dir}")
        print(f"  hard_negatives_to_mine:        {self.hard_negatives_to_mine}")
        print(f"  hard_neg_margin:               {self.hard_neg_margin}")
        print(f"  hard_neg_margin_type:          {self.hard_neg_margin_type}")
        print(f"  mining_batch_size:             {self.mining_batch_size}")
        print(f"  query_embedding_batch_size:    {self.query_embedding_batch_size}")
        print(f"  document_embedding_batch_size: {self.document_embedding_batch_size}")
        print(f"  corpus_chunk_size:             {self.corpus_chunk_size}")
        print(f"  load_embeddings_from_cache:    {self.load_embeddings_from_cache}")
        print(f"  use_negatives_from_file:       {self.use_negatives_from_file}")
        print("\nEmbedding configuration:")
        print(f"  query_prefix:                  '{self.query_prefix}'")
        print(f"  passage_prefix:                '{self.passage_prefix}'")
        print(f"  query_max_length:              {self.query_max_length}")
        print(f"  passage_max_length:            {self.passage_max_length}")
        print("\nModel (loaded directly from checkpoint):")
        print(f"  model_name_or_path:     {self.model_name_or_path}")
        print(f"  tokenizer_name_or_path: {self.tokenizer_name_or_path}")
        # Use cached metadata if available, otherwise get from model
        pooling = self._model_pooling if hasattr(self, "_model_pooling") else self.model.pooling
        l2_normalize = self._model_l2_normalize if hasattr(self, "_model_l2_normalize") else self.model.l2_normalize
        print(f"  pooling:                {pooling}")
        print(f"  l2_normalize:           {l2_normalize}")
        print("\nData:")
        print(f"  corpus_path:        {self.corpus_path}")
        print(f"  num_questions:      {len(self.questions)}")
        print(f"  num_documents:      {len(self.doc_to_idx)}")
        total_positives = sum(len(pos) for pos in self.pos_doc_indices)
        total_supplied_negs = sum(len(neg) for neg in self.supplied_neg_doc_indices)
        print(f"  total_positives:    {total_positives}")
        print(f"  total_supplied_neg: {total_supplied_negs}")
        print("\nDistributed environment:")
        print(f"  rank:       {self.dist_env.rank}")
        print(f"  world_size: {self.dist_env.world_size}")
        print(f"  backend:    {self.dist_env.backend}")
        print("=" * 60)
