import os
import pkg_resources
import pickle
from typing import Dict, Optional

from .config import config

import h5py
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from joblib import Memory


home_dir = os.path.expanduser('~')
cachedir = os.path.join(home_dir, '.cache', 'protac_degradation_predictor')
memory = Memory(cachedir, verbose=0)


@memory.cache
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
        embeddings_path = pkg_resources.resource_stream(__name__, 'data/uniprot2embedding.h5')
    protein2embedding = {}
    with h5py.File(embeddings_path, "r") as file:
        for sequence_id in file.keys():
            embedding = file[sequence_id][:]
            protein2embedding[sequence_id] = np.array(embedding)
    return protein2embedding


@memory.cache
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
        embeddings_path = pkg_resources.resource_stream(__name__, 'data/cell2embedding.pkl')
    with open(embeddings_path, 'rb') as f:
        cell2embedding = pickle.load(f)
    return cell2embedding


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