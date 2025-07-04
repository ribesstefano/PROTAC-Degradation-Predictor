import os
import pickle
from typing import Dict, Optional, List, Literal
from pathlib import Path
import requests
import logging
import functools

import gdown
import h5py
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from protac_degradation_predictor.config import config


home_dir = os.path.expanduser('~')
cachedir = os.path.join(home_dir, '.cache', 'protac_degradation_predictor')


def download_file(url: str, dest: Path, hash: Optional[str] = None):
    """ Download a file from a URL to a destination path.
    Args:
        url (str): The URL to download the file from.
        dest (Path): The destination path where the file will be saved.
    """
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


@functools.lru_cache()
def load_protein2embedding(
    embeddings_path: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """ Load the protein embeddings from a file.

    Args:
        embeddings_path (str): The path to the embeddings file.

    Returns:
        Dict[str, np.ndarray]: A dictionary of protein embeddings.
    """
    if embeddings_path is None:
        embeddings_path = Path(cachedir) / 'uniprot2embedding.h5'
        if not embeddings_path.exists():
            os.makedirs(embeddings_path.parent, exist_ok=True)
            download_file(
                url=config.uniprot2embedding_url,
                dest=embeddings_path,
            )
    protein2embedding = {}
    with h5py.File(embeddings_path, "r") as file:
        for sequence_id in file.keys():
            embedding = file[sequence_id][:]
            protein2embedding[sequence_id] = np.array(embedding)
    return protein2embedding


@functools.lru_cache()
def load_cell2embedding(
        embeddings_path: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """ Load the cell line embeddings from a file.
    
    Args:
        embeddings_path (str): The path to the embeddings file.
        
    Returns:
        Dict[str, np.ndarray]: A dictionary of cell line embeddings.
    """
    if embeddings_path is None:
        embeddings_path = Path(cachedir) / 'cell2embedding.pkl'
        if not embeddings_path.exists():
            os.makedirs(embeddings_path.parent, exist_ok=True)
            download_file(
                url=config.cell2embedding_url,
                dest=embeddings_path,
            )
    with open(embeddings_path, 'rb') as f:
        cell2embedding = pickle.load(f)
    return cell2embedding


@functools.lru_cache()
def load_curated_dataset() -> pd.DataFrame:
    """ Load the curated PROTAC dataset as described in the paper: https://arxiv.org/abs/2406.02637

    Returns:
        pd.DataFrame: The curated PROTAC dataset.
    """
    
    df_path = Path(cachedir) / 'PROTAC-Degradation-DB.csv'
    if not df_path.exists():
        os.makedirs(df_path.parent, exist_ok=True)
        download_file(
            url=config.curated_dataset_url,
            dest=df_path,
        )
    return pd.read_csv(df_path)


def avail_e3_ligases() -> List[str]:
    """ Get the available E3 ligases.
    
    Returns:
        List[str]: The available E3 ligases.
    """
    return list(config.e3_ligase2uniprot.keys())


def avail_cell_lines() -> List[str]:
    """ Get the available cell lines.
    
    Returns:
        List[str]: The available cell lines.
    """
    return list(load_cell2embedding().keys())


def avail_uniprots() -> List[str]:
    """ Get the available Uniprot IDs.
    
    Returns:
        List[str]: The available Uniprot IDs.
    """
    return list(load_protein2embedding().keys())


def get_fingerprint(smiles: str, morgan_fpgen = None) -> np.ndarray:
    """ Get the Morgan fingerprint of a molecule.
    
    Args:
        smiles (str): The SMILES string of the molecule.
        morgan_fpgen: The Morgan fingerprint generator.

    Returns:
        np.ndarray: The Morgan fingerprint.
    """
    if morgan_fpgen is None:
        morgan_fpgen = AllChem.GetMorganGenerator(
            radius=config.morgan_radius,
            fpSize=config.fingerprint_size,
            includeChirality=True,
        )
    return morgan_fpgen.GetFingerprint(Chem.MolFromSmiles(smiles))


def is_active(
        DC50: float,
        Dmax: float,
        pDC50_threshold: float = 7.0,
        Dmax_threshold: float = 0.8,
        oring: bool = False, # Deprecated
) -> bool:
    """ Check if a PROTAC is active based on DC50 and Dmax.	
    Args:
        DC50(float): DC50 in nM
        Dmax(float): Dmax in %
    Returns:
        bool: True if active, False if inactive, np.nan if either DC50 or Dmax is NaN
    """
    pDC50 = -np.log10(DC50 * 1e-9) if pd.notnull(DC50) else np.nan
    Dmax = Dmax / 100
    if pd.notnull(pDC50):
        if pDC50 < pDC50_threshold:
            return False
    if pd.notnull(Dmax):
        if Dmax < Dmax_threshold:
            return False
    if oring:
        if pd.notnull(pDC50):
            return True if pDC50 >= pDC50_threshold else False
        elif pd.notnull(Dmax):
            return True if Dmax >= Dmax_threshold else False
        else:
            return np.nan
    else:
        if pd.notnull(pDC50) and pd.notnull(Dmax):
            return True if pDC50 >= pDC50_threshold and Dmax >= Dmax_threshold else False
        else:
            return np.nan