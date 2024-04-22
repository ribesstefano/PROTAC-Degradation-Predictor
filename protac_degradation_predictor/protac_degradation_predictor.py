import pkg_resources
import logging

from .pytorch_models import PROTAC_Model, load_model
from .data_utils import (
    load_protein2embedding,
    load_cell2embedding,
    get_fingerprint,
)
from .config import config

import numpy as np
import torch
from torch import sigmoid


package_name = 'protac_degradation_predictor'


def get_protac_active_proba(
        protac_smiles: str,
        e3_ligase: str,
        target_uniprot: str,
        cell_line: str,
        device: str = 'cpu',
) -> bool:
    ckpt_path = pkg_resources.resource_stream(__name__, 'data/model.ckpt')
    model = load_model(ckpt_path).to(device)
    protein2embedding = load_protein2embedding()
    cell2embedding = load_cell2embedding()

    # Setup default embeddings
    if e3_ligase not in config.e3_ligase2uniprot:
        available_e3_ligases = ', '.join(list(config.e3_ligase2uniprot.keys()))
        logging.warning(f"The E3 ligase {e3_ligase} is not in the database. Using the default E3 ligase. Available E3 ligases are: {available_e3_ligases}")
    if target_uniprot not in protein2embedding:
        logging.warning(f"The target protein {target_uniprot} is not in the database. Using the default target protein.")
    if cell_line not in load_cell2embedding():
        logging.warning(f"The cell line {cell_line} is not in the database. Using the default cell line.")

    default_protein_emb = np.zeros(config.protein_embedding_size)
    default_cell_emb = np.zeros(config.cell_embedding_size)
    
    # Convert the E3 ligase to Uniprot ID
    e3_ligase_uniprot = config.e3_ligase2uniprot.get(e3_ligase, '')

    # Get the embeddings
    poi_emb = protein2embedding.get(target_uniprot, default_protein_emb)
    e3_emb = protein2embedding.get(e3_ligase_uniprot, default_protein_emb)
    cell_emb = cell2embedding.get(cell_line, default_cell_emb)
    smiles_emb = get_fingerprint(protac_smiles)

    # Convert to torch tensors
    poi_emb = torch.tensor(poi_emb).to(device)
    e3_emb = torch.tensor(e3_emb).to(device)
    cell_emb = torch.tensor(cell_emb).to(device)
    smiles_emb = torch.tensor(smiles_emb).to(device)

    return model(poi_emb, e3_emb, cell_emb, smiles_emb).item()


def is_protac_active(
        protac_smiles: str,
        e3_ligase: str,
        target_uniprot: str,
        cell_line: str,
        device: str = 'cpu',
        proba_threshold: float = 0.5,
) -> bool:
    """ Predict whether a PROTAC is active or not.
    
    Args:
        protac_smiles (str): The SMILES of the PROTAC.
        e3_ligase (str): The Uniprot ID of the E3 ligase.
        target_uniprot (str): The Uniprot ID of the target protein.
        cell_line (str): The cell line identifier.
        device (str): The device to run the model on.
        proba_threshold (float): The probability threshold.

    Returns:
        bool: Whether the PROTAC is active or not.
    """
    pred = get_protac_active_proba(
        protac_smiles,
        e3_ligase,
        target_uniprot,
        cell_line,
        device,
    )
    return sigmoid(pred) > proba_threshold