""" Handles molecular embeddings using RDKit and Hugging Face Transformers. """
import logging
from pathlib import Path
from typing import Optional, List, Union, Literal, Dict

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdFingerprintGenerator import FingerprintGenerator64
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from protac_degradation_predictor.config import config
from protac_degradation_predictor.data.embeddings.utils import EmbeddingMixin


class MolEmbedding(EmbeddingMixin):
    """ Class for handling molecular embeddings. """
    
    def __init__(
        self,
        embeddings_type: Literal["fingerprint", "transformer"] = "fingerprint",
        
        # Fingerprint-specific configurations
        radius: int = config.morgan_radius,
        fp_size: int = config.fingerprint_size,
        morgan_fpgen: Optional[FingerprintGenerator64] = None,
        
        # Transformer configurations
        pretrained_model: str = "ailab-bio/PROTAC-Splitter-Encoder",
        batch_size: int = 64,
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
        """ Initialize the MolEmbedding class.
        
        Args:
            embeddings_type: Type of embeddings to compute consistently for this instance
            radius: Radius for Morgan fingerprints (only used if embeddings_type="fingerprint")
            fp_size: Size of the Morgan fingerprints (only used if embeddings_type="fingerprint")
            morgan_fpgen: Predefined Morgan fingerprint generator (only used if embeddings_type="fingerprint")
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
            filename = f"mol_embeddings_{embeddings_type}.npz"
        
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
        
        # Initialize fingerprint-specific components
        self.radius = radius
        self.fp_size = fp_size
        self.morgan_fpgen = morgan_fpgen
        
        if embeddings_type == "fingerprint":
            if self.morgan_fpgen is None:
                self.morgan_fpgen = Chem.rdFingerprintGenerator.GetMorganGenerator(
                    radius=self.radius,
                    fpSize=self.fp_size,
                    includeChirality=True,
                )

    def encode(
        self,
        smiles: Union[str, List[str]],
        skip_existing: bool = True,
        update_cache: bool = False,
    ) -> Dict[str, Union[np.array, torch.Tensor]]:
        """ Encode SMILES strings into fingerprints or embeddings using the configured method.
        
        Args:
            smiles: SMILES string or list of SMILES strings
            skip_existing: Whether to skip existing embeddings in the cache
            update_cache: Whether to update the cache with the new embeddings
            
        Returns:
            Dict[str, Union[np.array, torch.Tensor]]: Encoded embeddings
        """
        if isinstance(smiles, str):
            smiles_list = [smiles]
        elif isinstance(smiles, list):
            smiles_list = smiles
        else:
            raise ValueError("Input smiles must be a string or a list of strings.")

        if skip_existing:
            smiles_to_encode = [s for s in smiles_list if s not in self.embeddings]
            smiles_encoded = {s: self.embeddings[s] for s in smiles_list if s in self.embeddings}
        else:
            smiles_to_encode = smiles_list

        if not smiles_to_encode:
            embeddings = {}
        else:
            embeddings = self._encode_smiles(smiles_to_encode)

        if skip_existing:
            all_embeddings = {**smiles_encoded, **embeddings}
            embeddings = {s: all_embeddings[s] for s in smiles_list}

        # Update instance embeddings
        if len(embeddings) > 0:
            self.embeddings.update(embeddings)

        if update_cache:
            self.save()

        return embeddings

    def _encode_smiles(self, smiles_list: List[str]) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """ Internal method to encode SMILES based on the configured embeddings_type. """
        if self.embeddings_type == "fingerprint":
            return self._encode_smiles_as_fingerprints(smiles_list)
        elif self.embeddings_type == "transformer":
            return self._encode_with_transformer(smiles_list)
        else:
            raise ValueError(f"Unsupported embeddings_type: {self.embeddings_type}")

    def _encode_smiles_as_fingerprints(self, smiles_list: List[str]) -> Dict[str, np.ndarray]:
        """ Encode SMILES as Morgan fingerprints. """
        fingerprints = {}
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES string: {smiles}")
            else:
                fp = self.morgan_fpgen.GetFingerprintAsNumPy(mol).astype(np.float32)
                fingerprints[smiles] = fp
        return fingerprints

    def _encode_with_transformer(self, smiles_list: List[str]) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """ Encode SMILES using transformer model. """
        return self.encode_with_transformer(
            strings=smiles_list,
            tokenizer=self.tokenizer,
            model=self.model,
            pretrained_model=self.pretrained_model,
            batch_size=self.batch_size,
            device=self.device,
            pooling=self.pooling,
            return_tensors=self.return_tensors,
            return_dict=True,
        )

    @staticmethod
    def encode_smiles_as_fingerprints(
        smiles: Union[str, List[str]],
        morgan_fpgen: Optional[FingerprintGenerator64] = None,
        radius: int = config.morgan_radius,
        fp_size: int = config.fingerprint_size,
    ) -> Dict[str, np.ndarray]:
        """ Static method to get the Morgan fingerprint of molecules.
        
        Args:
            smiles: The SMILES string(s) of the molecule(s)
            morgan_fpgen: The Morgan fingerprint generator
            radius: Radius for Morgan fingerprints
            fp_size: Size of the Morgan fingerprints

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping SMILES to fingerprints
        """
        if isinstance(smiles, str):
            smiles_list = [smiles]
        elif isinstance(smiles, list):
            smiles_list = smiles
        else:
            raise ValueError("Input smiles must be a string or a list of strings.")

        if morgan_fpgen is None:
            morgan_fpgen = Chem.rdFingerprintGenerator.GetMorganGenerator(
                radius=radius,
                fpSize=fp_size,
                includeChirality=True,
            )

        fingerprints = {}
        for smiles_str in smiles_list:
            mol = Chem.MolFromSmiles(smiles_str)
            if mol is None:
                raise ValueError(f"Invalid SMILES string: {smiles_str}")
            else:
                fp = morgan_fpgen.GetFingerprintAsNumPy(mol).astype(np.float32)
                fingerprints[smiles_str] = fp

        return fingerprints