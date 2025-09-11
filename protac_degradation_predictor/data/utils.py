""" Utility functions for handling saving and loading dictionaries in JSON,
pickle, or NPZ format.
"""
import json
import os
import pickle
import logging
from pathlib import Path
from typing import List, Union, Optional

import gdown
import numpy as np
import pandas as pd


def get_cache_dir() -> str:
    """Get the cache directory path and ensure it exists.
    
    Returns:
        str: Path to the cache directory
    """
    cache_dir = os.environ.get(
        "PROTAC_DEGRADATION_PREDICTOR_CACHE",
        os.path.join(os.path.expanduser('~'), '.cache', 'protac_degradation_predictor')
    )
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except PermissionError as e:
        # Fallback to a temporary directory
        import tempfile
        cache_dir = os.path.join(tempfile.gettempdir(), 'protac_degradation_predictor')
        os.makedirs(cache_dir, exist_ok=True)
        logging.warning(f"Permission denied creating cache directory. Using temporary directory: {cache_dir}")
    except Exception as e:
        logging.error(f"Failed to create cache directory {cache_dir}: {e}")
        raise
    return cache_dir

def download_file(url: str, dest: Path, hash: Optional[str] = None):
    """ Download a file from a URL to a destination path.
    Args:
        url (str): The URL to download the file from.
        dest (Path): The destination path where the file will be saved.
    """
    if not dest.parent.exists():
        os.makedirs(dest.parent, exist_ok=True)
        logging.debug(f"Created directory {dest.parent} for downloading file.")

    if not dest.exists():
        gdown.download(url, output=str(dest), quiet=False)
        logging.debug(f"Downloaded {url} to {dest}")

    if hash is not None:
        import hashlib
        sha256_hash = hashlib.sha256()
        with open(dest, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        if sha256_hash.hexdigest() != hash:
            raise ValueError(f"File {dest} does not match the expected hash {hash}.")

def save_dict(
    d: dict,
    filepath: str,
    indent: int = 4,
):
    """
    Save a dictionary to a file in JSON format.
    
    Args:
        d (dict): Dictionary to save.
        filepath (str): Path to the file where the dictionary will be saved.
        indent (int): Indentation level for JSON formatting. Default is 4.
        mode (str): File mode, either 'w' for text or 'wb' for binary. Default is 'w'.
    """
    if filepath.endswith('.json'):
        with open(filepath, 'w') as f:
            json.dump(d, f, indent=indent)
    elif filepath.endswith('.pkl'):
        with open(filepath, 'wb') as f:
            pickle.dump(d, f)
    else:
        raise ValueError(f'Unsupported file extension: {filepath}. Use .json or .pkl.')

def load_dict(filepath: str) -> Union[dict, List[dict]]:
    """
    Load a dictionary from a file in JSON or pickle format.
    
    Args:
        filepath (str): Path to the file from which the dictionary will be loaded.
        
    Returns:
        dict: The loaded dictionary. If the file does not exist, returns an empty dictionary.
    """
    if not os.path.exists(filepath):
        return {}
    if filepath.endswith('.json'):
        with open(filepath, 'r') as f:
            return json.load(f)
    elif filepath.endswith('.pkl'):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f'Unsupported file extension: {filepath}. Use .json or .pkl.')

def dmax_forward(dmax_percent, method="clipped_logit", clip=1e-3):
    """
    Forward transform from Dmax (%) to normalized scale.
    
    Parameters
    ----------
    dmax_percent : float or array-like
        Dmax values in % (0-100).
    method : {"clipped_logit","asin_sqrt"}
        Transform to apply.
    clip : float
        Clip proportions to [clip, 1-clip] for logit transform.
    
    Returns
    -------
    np.ndarray
        Transformed values.
    """
    if pd.isnull(dmax_percent):
        return np.nan
    
    arr = np.array(dmax_percent, dtype=float)
    mask = ~np.isnan(arr)
    p = arr / 100.0
    y = np.full_like(p, np.nan)

    if method == "clipped_logit":
        p_clip = np.clip(p, clip, 1 - clip)
        y[mask] = np.log(p_clip / (1 - p_clip))
    elif method == "asin_sqrt":
        p_clip = np.clip(p, 0, 1)  # arcsin-sqrt handles 0 and 1
        y[mask] = np.arcsin(np.sqrt(p_clip))
    else:
        raise ValueError("method must be 'clipped_logit' or 'asin_sqrt'")
    return y

def dmax_inverse(norm_values, method="clipped_logit", clip=1e-3):
    """
    Inverse transform from normalized scale back to Dmax (%).
    
    Parameters
    ----------
    norm_values : float or array-like
        Normalized values from forward transform.
    method : {"clipped_logit","asin_sqrt"}
        Transform used in forward pass.
    clip : float
        Same clipping parameter as in forward for logit.
    
    Returns
    -------
    np.ndarray
        Dmax values in % (0-100).
    """
    arr = np.array(norm_values, dtype=float)
    mask = ~np.isnan(arr)
    p = np.full_like(arr, np.nan)

    if method == "clipped_logit":
        p[mask] = 1.0 / (1.0 + np.exp(-arr[mask]))  # logistic
        # undo clipping range to [0,1] — optional:
        p = np.clip(p, clip, 1 - clip)
    elif method == "asin_sqrt":
        p[mask] = np.sin(arr[mask]) ** 2
    else:
        raise ValueError("method must be 'clipped_logit' or 'asin_sqrt'")
    
    return p * 100.0