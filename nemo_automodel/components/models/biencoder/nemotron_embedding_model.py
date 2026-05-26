#!/usr/bin/env python3
"""
Nemotron Embedding Model

Uses the 4b_nanov3_sft model to calculate text embeddings using the EOS token's
hidden state as the sentence representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Union, Optional
import numpy as np


class NemotronEmbeddingModel:
    """
    Embedding model that uses a Nemotron-based LLM to generate text embeddings.
    
    The embedding is computed by taking the hidden state at the EOS token position
    from the last layer of the model.
    """
    
    DEFAULT_MODEL_PATH = "/lustre/fsw/portfolios/llmservice/users/viraman/action_selection_evals/models/4b_nanov3_sft"
    
    def __init__(
        self,
        model_path: str = None,
        device: str = None,
        dtype: torch.dtype = torch.float16,
        normalize_embeddings: bool = True,
        pooling_strategy: str = "eos",  # "eos", "mean", "last"
        max_length: int = 4096,
    ):
        """
        Initialize the Nemotron embedding model.
        
        Args:
            model_path: Path to the model. Defaults to 4b_nanov3_sft.
            device: Device to run the model on. Defaults to cuda if available.
            dtype: Data type for the model. Defaults to float16.
            normalize_embeddings: Whether to L2-normalize embeddings. Defaults to True.
            pooling_strategy: How to pool token embeddings. Options: "eos", "mean", "last".
            max_length: Maximum sequence length. Defaults to 4096.
        """
        self.model_path = model_path or self.DEFAULT_MODEL_PATH
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.normalize_embeddings = normalize_embeddings
        self.pooling_strategy = pooling_strategy
        self.max_length = max_length
        
        print(f"Loading model from {self.model_path}...")
        self._load_model()
        print(f"Model loaded successfully on {self.device}")
    
    def _load_model(self):
        """Load the model and tokenizer."""
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        
        # Ensure EOS token is set
        if self.tokenizer.eos_token is None:
            self.tokenizer.eos_token = "</s>"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            device_map=self.device,
            trust_remote_code=True,
            output_hidden_states=True,  # Required to get hidden states
        )
        self.model.eval()
        
        # Get embedding dimension from model config
        self.embedding_dim = self.model.config.hidden_size
        print(f"Embedding dimension: {self.embedding_dim}")
    
    def _get_eos_positions(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Find the position of the EOS token for each sequence in the batch.
        If no EOS token, use the last non-padded position.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            eos_positions: Position of EOS/last token for each sequence [batch_size]
        """
        batch_size = input_ids.shape[0]
        eos_token_id = self.tokenizer.eos_token_id
        
        eos_positions = []
        for i in range(batch_size):
            # Find EOS token position
            eos_mask = (input_ids[i] == eos_token_id)
            if eos_mask.any():
                # Use the first EOS token position
                eos_pos = eos_mask.nonzero(as_tuple=True)[0][0].item()
            else:
                # Use last non-padded position
                eos_pos = attention_mask[i].sum().item() - 1
            eos_positions.append(eos_pos)
        
        return torch.tensor(eos_positions, device=input_ids.device)
    
    def _pool_embeddings(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pool token-level hidden states to get sequence embeddings.
        
        Args:
            hidden_states: Last layer hidden states [batch_size, seq_len, hidden_dim]
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            embeddings: Sequence embeddings [batch_size, hidden_dim]
        """
        if self.pooling_strategy == "eos":
            # Use hidden state at EOS token position
            eos_positions = self._get_eos_positions(input_ids, attention_mask)
            batch_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
            embeddings = hidden_states[batch_indices, eos_positions, :]
            
        elif self.pooling_strategy == "last":
            # Use hidden state at last non-padded position
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
            embeddings = hidden_states[batch_indices, seq_lengths, :]
            
        elif self.pooling_strategy == "mean":
            # Mean pooling over non-padded tokens
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
            embeddings = sum_embeddings / sum_mask
            
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
        
        return embeddings
    
    @torch.no_grad()
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 8,
        show_progress: bool = False,
        add_eos: bool = True,
    ) -> np.ndarray:
        """
        Encode texts into embeddings.
        
        Args:
            texts: Single text or list of texts to encode.
            batch_size: Batch size for encoding. Defaults to 8.
            show_progress: Whether to show progress bar. Defaults to False.
            add_eos: Whether to append EOS token to texts. Defaults to True.
            
        Returns:
            embeddings: Numpy array of embeddings [num_texts, hidden_dim]
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Optionally add EOS token to texts
        if add_eos:
            texts = [text + self.tokenizer.eos_token for text in texts]
        
        all_embeddings = []
        
        # Process in batches
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(range(0, len(texts), batch_size), desc="Encoding")
        else:
            iterator = range(0, len(texts), batch_size)
        
        for i in iterator:
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
                return_dict=True,
            )
            
            # Get last layer hidden states
            hidden_states = outputs.hidden_states[-1]
            
            # Pool to get sequence embeddings
            embeddings = self._pool_embeddings(
                hidden_states,
                inputs["input_ids"],
                inputs["attention_mask"],
            )
            
            # Normalize if requested
            if self.normalize_embeddings:
                embeddings = F.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.append(embeddings.cpu().float().numpy())
        
        return np.concatenate(all_embeddings, axis=0)
    
    def similarity(
        self,
        texts1: Union[str, List[str]],
        texts2: Union[str, List[str]],
    ) -> np.ndarray:
        """
        Compute cosine similarity between two sets of texts.
        
        Args:
            texts1: First set of texts.
            texts2: Second set of texts.
            
        Returns:
            similarities: Cosine similarity scores.
        """
        embeddings1 = self.encode(texts1)
        embeddings2 = self.encode(texts2)
        
        # Compute cosine similarity
        if embeddings1.ndim == 1:
            embeddings1 = embeddings1.reshape(1, -1)
        if embeddings2.ndim == 1:
            embeddings2 = embeddings2.reshape(1, -1)
        
        similarities = np.dot(embeddings1, embeddings2.T)
        return similarities


def main():
    """Example usage of the NemotronEmbeddingModel."""
    print("=" * 70)
    print("Nemotron Embedding Model Demo")
    print("=" * 70)
    
    # Initialize model
    model = NemotronEmbeddingModel(
        normalize_embeddings=True,
        pooling_strategy="eos",
    )
    
    # Example texts
    texts = [
        "The player should move left to avoid the enemy attack.",
        "Move to the left to dodge incoming damage.",
        "Cast fireball to deal damage to the goblin.",
        "Use health potion to restore HP.",
    ]
    
    print("\nEncoding example texts...")
    embeddings = model.encode(texts, show_progress=True)
    
    print(f"\nEmbedding shape: {embeddings.shape}")
    print(f"Embedding dtype: {embeddings.dtype}")
    
    # Compute pairwise similarities
    print("\nPairwise cosine similarities:")
    similarities = model.similarity(texts, texts)
    
    for i, text1 in enumerate(texts):
        for j, text2 in enumerate(texts):
            if i < j:
                print(f"  [{i}] vs [{j}]: {similarities[i, j]:.4f}")
                print(f"      '{text1[:50]}...'")
                print(f"      '{text2[:50]}...'")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
