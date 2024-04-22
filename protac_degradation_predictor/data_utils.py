import os
import pkg_resources
import pickle
from typing import Dict

from config import config

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
def load_protein2embedding() -> Dict[str, np.ndarray]:
    embeddings_path = pkg_resources.resource_stream(__name__, 'data/uniprot2embedding.h5')
    protein2embedding = {}
    with h5py.File(embeddings_path, "r") as file:
        for sequence_id in file.keys():
            embedding = file[sequence_id][:]
            protein2embedding[sequence_id] = np.array(embedding)
    return protein2embedding


@memory.cache
def load_cell2embedding() -> Dict[str, np.ndarray]:
    embeddings_path = pkg_resources.resource_stream(__name__, 'data/cell2embedding.pkl')
    with open(embeddings_path, 'rb') as f:
        cell2embedding = pickle.load(f)
    return cell2embedding


def get_fingerprint(smiles: str) -> np.ndarray:
    morgan_fpgen = AllChem.GetMorganGenerator(
        radius=config.morgan_radius,
        fpSize=config.fingerprint_size,
        includeChirality=True,
    )
    return morgan_fpgen.GetFingerprint(Chem.MolFromSmiles(smiles))