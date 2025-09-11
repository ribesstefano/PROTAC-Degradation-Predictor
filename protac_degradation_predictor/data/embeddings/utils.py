""" """
import logging
from pathlib import Path
from typing import Dict, Optional, Union, List, Literal

import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer

from protac_degradation_predictor.data.utils import get_cache_dir


class EmbeddingMixin(object):
    """ Wrapper class for handling embeddings. """
    
    def __init__(
            self,
            embeddings: Optional[Union[Dict[str, np.ndarray], np.ndarray]] = None,
            model: Optional[Union[AutoModel, str]] = None,
            tokenizer: Optional[Union[AutoTokenizer, str]] = None,
            load_from_cache: bool = True,
            filename: Optional[Union[Path, str]] = None,
            cache_dir: Optional[Union[Path, str]] = None,
    ):
        if isinstance(embeddings, dict):
            self.embeddings = embeddings
        elif isinstance(embeddings, np.ndarray):
            self.embeddings = {str(i): emb for i, emb in enumerate(embeddings)}
        else:
            self.embeddings = {}

        self.filename = filename
        self.cache_dir = cache_dir
        if load_from_cache and filename is not None:
            self.load(filename=filename, cache_dir=cache_dir)

        self.model = None
        if model is not None:
            if isinstance(model, str):
                self.model = AutoModel.from_pretrained(model)
            else:
                self.model = model
            self.model.eval()  # Set the model to evaluation mode

        self.tokenizer = None
        if tokenizer is not None:
            if isinstance(tokenizer, str):
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
            else:
                self.tokenizer = tokenizer

    def __getitem__(self, key: str) -> np.ndarray:
        return self.embeddings[key]
    
    def __contains__(self, key: str) -> bool:
        return key in self.embeddings

    def __len__(self) -> int:
        return len(self.embeddings)

    def shape(self) -> Union[tuple, List[tuple]]:
        """ Get the shape of the embeddings.
        
        Returns:
            Union[tuple, List[tuple]]: Shape of the embeddings. If multiple embeddings are present, returns a list of shapes.
        """
        if isinstance(self.embeddings, dict):
            shapes = {emb.shape for emb in self.embeddings.values()}
            if len(shapes) == 1:
                shape = shapes.pop()
                # If all but one dimension are 1, flatten the shape.
                # Example: (1, 128) -> (128,) or (1, 1, 128) -> (128,)
                if all(dim == 1 for dim in shape[:-1]):
                    logging.debug(f"Flattening shape {shape}")
                    return (shape[-1],)
                # Example: (128, 1) -> (128,) or (128, 1, 1) -> (128,)
                elif all(dim == 1 for dim in shape[1:]):
                    return (shape[0],)
                return shape
            else:
                logging.debug(f"Multiple shapes found in embeddings: {shapes}. Returning as a list.")
                raise ValueError("Embeddings must all have the same shape.")
        elif isinstance(self.embeddings, np.ndarray):
            return self.embeddings.shape
        else:
            raise ValueError("Embeddings must be a dictionary or a numpy array.")

    def save(
            self,
            filename: Optional[Union[Path, str]] = None,
            cache_dir: Optional[str] = None,
    ):
        """ Save embeddings to a file.
        
        Args:
            filename (Optional[Union[Path, str]]): Name of the file to save the embeddings. If None, uses the default filename defined in the class.
                If the filename is not set, it will use the default filename "embeddings.npz".
            cache_dir (Optional[str]): Directory to save the embeddings. If None, uses the default cache directory.
                If the cache directory is not set, it will use the default cache directory defined as environment variable PROTAC_DEGRADATION_PREDICTOR_CACHE, if set, or: ~/.cache/protac_degradation_predictor

        """
        if cache_dir is None:
            if self.cache_dir is None:
                cache_dir = get_cache_dir()
            else:
                cache_dir = self.cache_dir

        if filename is None:
            filename = self.filename if self.filename is not None else "embeddings.npz"

        filepath = Path(cache_dir) / filename

        if not filepath.parent.exists():
            filepath.parent.mkdir(parents=True, exist_ok=True)
            logging.debug(f"Created directory {filepath.parent} for saving embeddings.")

        embeddings = {
            k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in self.embeddings.items()
        }
        np.savez(filepath, **embeddings)
        logging.info(f"Embeddings saved to {cache_dir}/{filename}")
    
    def load(self, filename: Optional[Union[Path, str]] = None, cache_dir: Optional[str] = None):
        """ Load embeddings from a file. """
        cache_dir = get_cache_dir() if cache_dir is None else cache_dir
        filepath = Path(cache_dir) / filename if filename else Path(cache_dir) / self.filename

        if not filepath.exists():
            logging.warning(f"File {filepath} does not exist. Returning empty embeddings.")
            return {}

        loaded_embeddings = np.load(filepath, allow_pickle=True)
        self.embeddings = {k: v for k, v in loaded_embeddings.items()}
        logging.info(f"Embeddings loaded from {filepath}")

    def to_numpy(self) -> np.ndarray:
        """ Convert the embeddings to a single numpy array. """
        return np.stack(list(self.embeddings.values()), axis=0)
    
    def to_tensor(self) -> Dict[str, torch.Tensor]:
        """ Convert the embeddings to a single tensor. """
        return torch.stack(list(self.embeddings.values()), dim=0)
    
    def preprocess(
        self,
        op: Literal["standardize", "normalize", "minmax"] = "standardize",
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """ Preprocess the embeddings using a specified operation. The operation
        won't update the internal embeddings dictionary nor save it to cache.
        
        Args:
            op (Literal["standardize", "normalize", "minmax"]): The preprocessing operation to apply.
            **kwargs: Additional keyword arguments for the preprocessing method.
        
        Returns:
            None: Updates the internal embeddings dictionary.
        """
        if op == "standardize":
            scaler = StandardScaler(**kwargs)
        elif op == "normalize":
            scaler = Normalizer(**kwargs)
        elif op == "minmax":
            scaler = MinMaxScaler(**kwargs)
        else:
            raise ValueError(f"Unsupported operation: {op}")

        embeddings = self.to_numpy()
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        scaled_embeddings = scaler.fit_transform(embeddings)
        return {k: v for k, v in zip(self.embeddings.keys(), scaled_embeddings)}

    def encode(self, **kwargs) -> Union[Dict[str, np.ndarray], np.ndarray]:
        """ Encode a list of strings into embeddings.
        
        Args:
            strings (Union[str, List[str]]): A single string or a list of strings to encode.
            **kwargs: Additional keyword arguments for the encoding method.
        
        Returns:
            Union[Dict[str, np.ndarray], np.ndarray]: Encoded embeddings.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def update(self, d: Dict[str, np.ndarray]) -> None:
        """ Update the embeddings with a new dictionary of embeddings.
        
        Args:
            d (Dict[str, np.ndarray]): Dictionary of new embeddings to update.
        
        Returns:
            None: Updates the internal embeddings dictionary.
        """
        if not isinstance(d, dict):
            raise ValueError("Input must be a dictionary.")
        
        for key, value in d.items():
            if not isinstance(value, np.ndarray):
                raise ValueError(f"Value for key '{key}' must be a NumPy array.")
            self.embeddings[key] = value

    def encode_update(self, to_encode: Union[str, List[str]], **kwargs) -> None:
        """ Run encoding, then update the embeddings with new embeddings.
        
        Args:
            to_encode (Union[str, List[str]]): A single string or a list of strings to encode.
            **kwargs: Additional keyword arguments for the encoding method.
        
        Returns:
            None: Updates the internal embeddings dictionary.
        """
        self.update(self.encode(to_encode, **kwargs))

    @staticmethod
    def encode_with_transformer(
            strings: Union[str, List[str]],
            tokenizer: Optional[AutoTokenizer] = None,
            model: Optional[AutoModel] = None,
            pretrained_model: str = "ailab-bio/PROTAC-Splitter-Encoder",
            batch_size: int = 64,
            device: Union[int, str] = "cpu",
            pooling: Literal["cls", "mean", "sum", "max", "mean_sqrt_len"] = "sum",
            return_tensors: Literal["pt", "np"] = "np",
            return_dict: bool = False,
    ) -> Union[torch.Tensor, np.ndarray]:
        """ Encode a list of strings strings into embeddings using a pre-trained model.
        
        Args:
            strings (List[str]): List of strings strings to encode.
            tokenizer (AutoTokenizer): Tokenizer for the model.
            model (AutoModel): Pre-trained model to use for encoding.
            batch_size (int): Batch size for encoding.
            device (Union[int, str]): Device to run the model on ("cpu" or "cuda").
            pooling (Literal["cls", "mean", "sum", "max", "mean_sqrt_len"]): Pooling method to use for pooling the embeddings along the sequence length dimension.
            
        Returns:
            torch.Tensor: Tensor of shape (num_strings, embedding_dim) containing the embeddings.
        """
        if isinstance(strings, str):
            strings = [strings]
        elif isinstance(strings, list):
            strings = strings
        else:
            raise ValueError("Input strings must be a string or a list of strings.")

        # Load the pre-trained model if not provided
        if tokenizer is None or model is None:
            logging.warning(f"Loading pre-trained model {pretrained_model} for encoding strings.")
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
            model = AutoModel.from_pretrained(pretrained_model)

        # Move the model to the specified device, then process the strings batches
        model = model.to(device)
        embeddings = []
        for i in range(0, len(strings), batch_size):
            batch = strings[i:i+batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                if pooling == "cls":
                    # Use the [CLS] token embedding as the representation
                    batch_embeds = outputs.last_hidden_state[:, 0, :].cpu()
                elif pooling == "mean":
                    # Pool the embeddings by averaging
                    batch_embeds = outputs.last_hidden_state.mean(dim=1).cpu()
                elif pooling == "sum":
                    # Pool the embeddings by summing
                    batch_embeds = outputs.last_hidden_state.sum(dim=1).cpu()
                elif pooling == "max":
                    # Pool the embeddings by taking the max
                    batch_embeds = outputs.last_hidden_state.max(dim=1).values.cpu()
                elif pooling == "mean_sqrt_len":
                    # Pool the embeddings by averaging and scaling by the square root of the sequence length
                    seq_lengths = inputs["input_ids"].ne(tokenizer.pad_token_id).sum(dim=1, keepdim=True).float()
                    batch_embeds = (outputs.last_hidden_state.sum(dim=1) / seq_lengths.sqrt()).cpu()
                else:
                    raise ValueError(f"Unsupported pooling method: {pooling}")
                embeddings.append(batch_embeds)

        embeddings = torch.cat(embeddings, dim=0)
        if return_tensors == "np":
            embeddings = embeddings.numpy()

        if return_dict:
            return {s: emb for s, emb in zip(strings, embeddings)}

        return embeddings