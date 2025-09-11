""" Handles protein embeddings using SciKit-Learn and Hugging Face Transformers. """
from pathlib import Path
from typing import Optional, List, Union, Literal, Dict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import CountVectorizer

from protac_degradation_predictor.data.embeddings.utils import EmbeddingMixin


class ProteinEmbedding(EmbeddingMixin):
    """ Class for handling protein embeddings. """
    
    def __init__(
        self,
        embeddings_type: Literal["amino_acid_count", "transformer", "esm", "boltz2_s", "boltz2_z", "boltz_s_z"] = "esm",
        # Embedding-specific configurations
        count_vect_kwargs: Optional[dict] = None,
        boltz_output_dir: Optional[Union[Path, str]] = None,
        
        # Transformer configurations
        pretrained_model: str = "facebook/esm2_t6_8M_UR50D",
        batch_size: int = 16,
        device: Union[int, str] = "cpu",
        pooling: Literal["cls", "mean", "sum", "max", "mean_sqrt_len"] = "sum",
        return_tensors: Literal["pt", "np"] = "np",
        
        # EmbeddingMixin parameters
        embeddings: Optional[Union[Dict[str, np.ndarray], np.ndarray]] = None,
        model: Optional[Union[AutoModel, str]] = None,
        tokenizer: Optional[Union[AutoTokenizer, str]] = None,
        load_from_cache: bool = False,
        filename: Optional[Union[Path, str]] = None,
        cache_dir: Optional[Union[Path, str]] = None,
    ):
        """ Initialize the ProteinEmbedding class.
        
        Args:
            embeddings_type: Type of embeddings to compute consistently for this instance
            count_vect_kwargs: Parameters for CountVectorizer (only used if embeddings_type="amino_acid_count")
            boltz_output_dir: Directory containing Boltz2 embeddings files (only used for boltz2 types)
            pretrained_model: Name of the pre-trained model to use if tokenizer or model is None
            batch_size: Batch size for transformer encoding
            device: Device to run the model on ("cpu" or "cuda")
            pooling: Pooling method for transformer embeddings
            return_tensors: Format of the returned tensors
            embeddings: Precomputed embeddings or fingerprints
            model: Pre-trained transformer model for embeddings
            tokenizer: Tokenizer for the transformer model
            load_from_cache: Whether to load embeddings from cache
            filename: Path to the file containing embeddings
            cache_dir: Directory to store cached embeddings
        """
        # Set default filename based on embeddings_type if not provided
        if filename is None:
            filename = f"protein_embeddings_{embeddings_type}.npz"
        
        super().__init__(
            embeddings=embeddings,
            model=model,
            tokenizer=tokenizer,
            load_from_cache=load_from_cache,
            filename=filename,
            cache_dir=cache_dir,
        )

        # Store embedding configuration
        self.embeddings_type = embeddings_type
        self.pretrained_model = pretrained_model
        self.batch_size = batch_size
        self.device = device
        self.pooling = pooling
        self.return_tensors = return_tensors

        # Initialize type-specific components
        self.countvec = None
        if embeddings_type == "amino_acid_count":
            if count_vect_kwargs is None:
                raise ValueError("count_vect_kwargs must be provided for amino_acid_count embeddings")
            self.countvec = CountVectorizer(**count_vect_kwargs)
        
        if embeddings_type in ["boltz2_s", "boltz2_z"]:
            if boltz_output_dir is None:
                raise ValueError("boltz_output_dir must be provided for Boltz2 embeddings")
            self.boltz_output_dir = Path(boltz_output_dir)

    def encode(
        self,
        sequences: Union[str, List[str]],
        skip_existing: bool = True,
        update_cache: bool = False,
    ) -> Dict[str, Union[np.array, torch.Tensor]]:
        """ Encode sequences strings into fingerprints or embeddings using the configured method.
        
        Args:
            sequences: Sequence string or list of sequences strings
            skip_existing: Whether to skip sequences that are already in the embeddings
            update_cache: Whether to update the cache with new embeddings

        Returns:
            Dict[str, Union[np.array, torch.Tensor]]: Encoded embeddings
        """
        if isinstance(sequences, str):
            seq_list = [sequences]
        elif isinstance(sequences, list):
            seq_list = sequences
        else:
            raise ValueError("Input sequences must be a string or a list of strings.")

        if skip_existing:
            seq_to_encode = [s for s in seq_list if s not in self.embeddings]
            seq_encoded = {s: self.embeddings[s] for s in seq_list if s in self.embeddings}
        else:
            seq_to_encode = seq_list

        if not seq_to_encode:
            embeddings = {}
        else:
            embeddings = self._encode_sequences(seq_to_encode)

        if skip_existing:
            all_embeddings = {**seq_encoded, **embeddings}
            embeddings = {s: all_embeddings[s] for s in seq_list}

        # Update instance embeddings
        if len(embeddings) > 0:
            self.embeddings.update(embeddings)

        if update_cache:
            self.save()

        return embeddings

    def _encode_sequences(self, sequences: List[str]) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """ Internal method to encode sequences based on the configured embeddings_type. """
        if self.embeddings_type == "amino_acid_count":
            return self._encode_amino_acid_count(sequences)
        elif self.embeddings_type in ["transformer", "esm"]:
            return self._encode_with_transformer(sequences)
        elif self.embeddings_type in ["boltz2_s", "boltz2_z"]:
            return self._encode_boltz2(sequences)
        else:
            raise ValueError(f"Unsupported embeddings_type: {self.embeddings_type}")

    def _encode_amino_acid_count(self, sequences: List[str]) -> Dict[str, np.ndarray]:
        """ Encode sequences using amino acid counting. """
        # Reshape for CountVectorizer
        embeddings = self.countvec.fit_transform(sequences).toarray()
        return {s: e for s, e in zip(sequences, embeddings)}

    def _encode_with_transformer(self, sequences: List[str]) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """ Encode sequences using transformer model. """
        return self.encode_with_transformer(
            strings=sequences,
            tokenizer=self.tokenizer,
            model=self.model,
            pretrained_model=self.pretrained_model,
            batch_size=self.batch_size,
            device=self.device,
            pooling=self.pooling,
            return_tensors=self.return_tensors,
            return_dict=True,
        )

    def _encode_boltz2(self, sequences: List[str]) -> Dict[str, np.ndarray]:
        """ Encode sequences using Boltz2 embeddings. """
        embeddings = {}
        for seq_id in sequences:
            emb_file = self.boltz_output_dir / f"boltz_results_{seq_id}" / "predictions" / seq_id / f"embeddings_{seq_id}.npz"
            if not emb_file.exists():
                raise FileNotFoundError(f"Boltz2 embeddings file {emb_file} not found.")
            
            data = np.load(emb_file)
            s = data['s']
            z = data['z']

            if self.embeddings_type == "boltz2_s":
                embeddings[seq_id] = self.pool_boltz_embeddings(s=s, pooling=self.pooling)
            elif self.embeddings_type == "boltz2_z":
                embeddings[seq_id] = self.pool_boltz_embeddings(z=z, pooling=self.pooling, z_pooling="flatten")
        
        return embeddings

    @staticmethod
    def pool_boltz_embeddings(
            s: Optional[np.ndarray] = None,
            z: Optional[np.ndarray] = None,
            pooling: Literal["mean", "sum", "max", "mean_sqrt_len"] = "sum",
            z_pooling: Literal["none", "flatten", "sum_axis0", "sum_axis1"] = "flatten",
    ) -> np.ndarray:
        """ Pool Boltz2 embeddings. The s embeddings are of shape (L, D) and z embeddings are of shape (L, D, D).
        
        Args:
            s (Optional[np.ndarray]): Boltz2 S embeddings.
            z (Optional[np.ndarray]): Boltz2 Z embeddings.
            pooling (Literal["mean", "sum", "max", "mean_sqrt_len"]): Pooling method for reducing over the L dimension.
            z_pooling (Literal["none", "flatten", "sum_axis0", "sum_axis1"]): Pooling method for reducing over one D dimension of the z embeddings.

        Returns:
            np.ndarray: Pooled embeddings.
        """
        
        # Example shapes:
        # s: (1, 350, 384)
        # z: (1, 350, 350, 128)
        # s: (1, 498, 384)
        # z: (1, 498, 498, 128)

        if s is not None and z is not None:
            raise NotImplementedError("Both 's' and 'z' cannot be provided at the same time. Choose one.")
        elif s is not None:
            s = s[0]
            # s is of shape (L, D)
            if pooling == "mean":
                return np.mean(s, axis=0)
            elif pooling == "sum":
                return np.sum(s, axis=0)
            elif pooling == "max":
                return np.max(s, axis=0)
            elif pooling == "mean_sqrt_len":
                return np.mean(s, axis=0) / np.sqrt(s.shape[0])
            else:
                raise ValueError(f"Unsupported pooling method: {pooling}. Choose from 'mean', 'sum', 'max', or 'mean_sqrt_len'.")
        elif z is not None:
            z = z[0]
            # z is of shape (L, L, D)
            if pooling == "mean":
                emb = np.mean(z, axis=(0, 1))
            elif pooling == "sum":
                emb = np.sum(z, axis=(0, 1))
            elif pooling == "max":
                emb = np.max(z, axis=(0, 1))
            elif pooling == "mean_sqrt_len":
                emb = np.mean(z, axis=(0, 1)) / np.sqrt(z.shape[0])

            if z_pooling == "none":
                return emb
            elif z_pooling == "flatten":
                return emb.flatten()
            elif z_pooling == "sum_axis0":
                return np.sum(emb, axis=0)
            elif z_pooling == "sum_axis1":
                return np.sum(emb, axis=1)
        else:
            raise ValueError("Either 's' or 'z' must be provided.")