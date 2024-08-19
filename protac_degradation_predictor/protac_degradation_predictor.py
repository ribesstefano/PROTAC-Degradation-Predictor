import os
import pkg_resources
import logging
from typing import List, Literal, Dict

from .pytorch_models import PROTAC_Model, load_model
from .data_utils import (
    load_protein2embedding,
    load_cell2embedding,
    get_fingerprint,
    load_curated_dataset,
)
from .config import config

import numpy as np
import torch
from torch import sigmoid
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer


def get_protac_active_proba(
        protac_smiles: str | List[str],
        e3_ligase: str | List[str],
        target_uniprot: str | List[str],
        cell_line: str | List[str],
        device: Literal['cpu', 'cuda'] = 'cpu',
        use_models_from_cv: bool = False,
        use_xgboost_models: bool = False,
        study_type: Literal['standard', 'similarity', 'target'] = 'standard',
) -> Dict[str, np.ndarray]:
    """ Predict the probability of a PROTAC being active.

    Args:
        protac_smiles (str | List[str]): The SMILES of the PROTAC.
        e3_ligase (str | List[str]): The Uniprot ID of the E3 ligase.
        target_uniprot (str | List[str]): The Uniprot ID of the target protein.
        cell_line (str | List[str]): The cell line identifier.
        device (str): The device to run the model on.
        use_models_from_cv (bool): Whether to use the models from cross-validation.
        use_xgb_models (bool): Whether to use the XGBoost models.
        study_type (str): Use models trained on the specified study. Options are 'standard', 'similarity', 'target'.

    Returns:
        Dict[str, np.ndarray]: The predictions of the model. The dictionary contains the following: 'preds', 'mean', 'majority_vote'. The 'preds' key contains the predictions of all models with shape: (n_models, batch_size), 'mean' contains the mean prediction, and 'majority_vote' contains the majority vote.
    """
    # Check that the study type is valid
    if study_type not in ['standard', 'similarity', 'target']:
        raise ValueError(f"Invalid study type: {study_type}. Options are 'standard', 'similarity', 'target'.")

    # Check that the device is valid
    if device not in ['cpu', 'cuda']:
        raise ValueError(f"Invalid device: {device}. Options are 'cpu', 'cuda'.")
    
    # Check that if any the models input is a list, all inputs are lists
    model_inputs = [protac_smiles, e3_ligase, target_uniprot, cell_line]
    if any(isinstance(i, list) for i in model_inputs):
        if not all(isinstance(i, list) for i in model_inputs):
            raise ValueError("All model inputs must be lists if one of the inputs is a list.")

    # Load all required models in pkg_resources
    device = torch.device(device)
    models = {}
    model_to_load = 'best_model' if not use_models_from_cv else 'cv_model'
    for model_filename in pkg_resources.resource_listdir(__name__, 'models'):
        if model_to_load not in model_filename:
            continue
        if study_type not in model_filename:
            continue
        if not use_xgboost_models:
            if 'xgboost' not in model_filename:
                ckpt_path = pkg_resources.resource_filename(__name__, f'models/{model_filename}')
                models[ckpt_path] = load_model(ckpt_path).to(device)
        else:
            if 'xgboost' in model_filename:
                json_path = pkg_resources.resource_filename(__name__, f'models/{model_filename}')
                models[json_path] = xgb.Booster()
                models[json_path].load_model(json_path)

    protein2embedding = load_protein2embedding()
    cell2embedding = load_cell2embedding()

    # Get the dimension of the embeddings from the first np.array in the dictionary
    protein_embedding_size = next(iter(protein2embedding.values())).shape[0]
    cell_embedding_size = next(iter(cell2embedding.values())).shape[0]
    # Setup default embeddings
    default_protein_emb = np.zeros(protein_embedding_size)
    default_cell_emb = np.zeros(cell_embedding_size)

    # Check if any model name contains cellsonehot, if so, get onehot encoding
    cell2onehot = None
    if any('cellsonehot' in model_name for model_name in models.keys()):
        onehotenc = OneHotEncoder(sparse_output=False)
        cell_embeddings = onehotenc.fit_transform(
            np.array(list(cell2embedding.keys())).reshape(-1, 1)
        )
        cell2onehot = {k: v for k, v in zip(cell2embedding.keys(), cell_embeddings)}
    
    # Check if any of the model names contain aminoacidcnt, if so, get the CountVectorizer
    protein2aacnt = None
    if any('aminoacidcnt' in model_name for model_name in models.keys()):
        # Create a new protein2embedding dictionary with amino acid sequence
        protac_df = load_curated_dataset()
        # Create the dictionary mapping 'Uniprot' to 'POI Sequence'
        protein2aacnt = protac_df.set_index('Uniprot')['POI Sequence'].to_dict()
        # Create the dictionary mapping 'E3 Ligase Uniprot' to 'E3 Ligase Sequence'
        e32seq = protac_df.set_index('E3 Ligase Uniprot')['E3 Ligase Sequence'].to_dict()
        # Merge the two dictionaries into a new protein2aacnt dictionary
        protein2aacnt.update(e32seq)

        # Get count vectorized embeddings for proteins
        # NOTE: Check that the protein2aacnt is a dictionary of strings
        if not all(isinstance(k, str) for k in protein2aacnt.keys()):
            raise ValueError("All keys in `protein2aacnt` must be strings.")
        countvec = CountVectorizer(ngram_range=(1, 1), analyzer='char')
        protein_embeddings = countvec.fit_transform(
            list(protein2aacnt.keys())
        ).toarray()
        protein2aacnt = {k: v for k, v in zip(protein2aacnt.keys(), protein_embeddings)}

    # Convert the E3 ligase to Uniprot ID
    if isinstance(e3_ligase, list):
        e3_ligase_uniprot = [config.e3_ligase2uniprot.get(e3, '') for e3 in e3_ligase]
    else:
        e3_ligase_uniprot = config.e3_ligase2uniprot.get(e3_ligase, '')

    # Get the embeddings for the PROTAC, E3 ligase, target protein, and cell line
    # Check if the input is a list or a single string, in the latter case,
    # convert to a list to create a batch of size 1, len(list) otherwise.
    if isinstance(protac_smiles, list):
        # TODO: Add warning on missing entries?
        smiles_emb = [get_fingerprint(s) for s in protac_smiles]
        cell_emb = [cell2embedding.get(c, default_cell_emb) for c in cell_line]
        e3_emb = [protein2embedding.get(e3, default_protein_emb) for e3 in e3_ligase_uniprot]
        poi_emb = [protein2embedding.get(t, default_protein_emb) for t in target_uniprot]
        # Convert to one-hot encoded cell embeddings if necessary
        if cell2onehot is not None:
            cell_onehot = [cell2onehot.get(c, default_cell_emb) for c in cell_line]
        # Convert to amino acid count embeddings if necessary
        if protein2aacnt is not None:
            poi_aacnt = [protein2aacnt.get(t, default_protein_emb) for t in target_uniprot]
            e3_aacnt = [protein2aacnt.get(e3, default_protein_emb) for e3 in e3_ligase_uniprot]
    else:
        if e3_ligase not in config.e3_ligase2uniprot:
            available_e3_ligases = ', '.join(list(config.e3_ligase2uniprot.keys()))
            logging.warning(f"The E3 ligase {e3_ligase} is not in the database. Using the default E3 ligase. Available E3 ligases are: {available_e3_ligases}")
        if target_uniprot not in protein2embedding:
            logging.warning(f"The target protein {target_uniprot} is not in the database. Using the default target protein.")
        if cell_line not in cell2embedding:
            logging.warning(f"The cell line {cell_line} is not in the database. Using the default cell line.")
        smiles_emb = [get_fingerprint(protac_smiles)]
        cell_emb = [cell2embedding.get(cell_line, default_cell_emb)]
        poi_emb = [protein2embedding.get(target_uniprot, default_protein_emb)]
        e3_emb = [protein2embedding.get(e3_ligase_uniprot, default_protein_emb)]
        # Convert to one-hot encoded cell embeddings if necessary
        if cell2onehot is not None:
            cell_onehot = [cell2onehot.get(cell_line, default_cell_emb)]
        # Convert to amino acid count embeddings if necessary
        if protein2aacnt is not None:
            poi_aacnt = [protein2aacnt.get(target_uniprot, default_protein_emb)]
            e3_aacnt = [protein2aacnt.get(e3_ligase_uniprot, default_protein_emb)]

    # Convert to numpy arrays
    smiles_emb = np.array(smiles_emb)
    cell_emb = np.array(cell_emb)
    poi_emb = np.array(poi_emb)
    e3_emb = np.array(e3_emb)
    if cell2onehot is not None:
        cell_onehot = np.array(cell_onehot)
    if protein2aacnt is not None:
        poi_aacnt = np.array(poi_aacnt)
        e3_aacnt = np.array(e3_aacnt)

    # Convert to torch tensors
    if not use_xgboost_models:
        smiles_emb = torch.tensor(smiles_emb).float().to(device)
        cell_emb = torch.tensor(cell_emb).to(device)
        poi_emb = torch.tensor(poi_emb).to(device)
        e3_emb = torch.tensor(e3_emb).to(device)
        if cell2onehot is not None:
            cell_onehot = torch.tensor(cell_onehot).float().to(device)
        if protein2aacnt is not None:
            poi_aacnt = torch.tensor(poi_aacnt).float().to(device)
            e3_aacnt = torch.tensor(e3_aacnt).float().to(device)
    
    # Average the predictions of all models
    preds = {}
    for ckpt_path, model in models.items():
        # Get the last part of the path
        ckpt_path = os.path.basename(ckpt_path)
        if not use_xgboost_models:
            pred = model(
                poi_emb if 'aminoacidcnt' not in ckpt_path else poi_aacnt,
                e3_emb if 'aminoacidcnt' not in ckpt_path else e3_aacnt,
                cell_emb if 'cellsonehot' not in ckpt_path else cell_onehot,
                smiles_emb,
                prescaled_embeddings=False, # Normalization performed by the model
            )
            preds[ckpt_path] = sigmoid(pred).detach().cpu().numpy().flatten()
        else:
            X = np.hstack([smiles_emb, poi_emb, e3_emb, cell_emb])
            pred = model.inplace_predict(X)
            preds[ckpt_path] = pred

    # NOTE: The predictions array has shape: (n_models, batch_size)
    preds = np.array(list(preds.values()))
    mean_preds = np.mean(preds, axis=0)
    # Return a single value if not list as input
    mean_preds = mean_preds if isinstance(protac_smiles, list) else mean_preds[0]
    
    return {
        'preds': preds,
        'mean': mean_preds,
        'majority_vote': mean_preds > 0.5,
    }


def is_protac_active(
        protac_smiles: str | List[str],
        e3_ligase: str | List[str],
        target_uniprot: str | List[str],
        cell_line: str | List[str],
        device: str = 'cpu',
        proba_threshold: float = 0.5,
        use_majority_vote: bool = False,
        use_models_from_cv: bool = False,
        use_xgboost_models: bool = False,
        study_type: Literal['standard', 'similarity', 'target'] = 'standard',
) -> bool:
    """ Predict whether a PROTAC is active or not.
    
    Args:
        protac_smiles (str): The SMILES of the PROTAC.
        e3_ligase (str): The Uniprot ID of the E3 ligase.
        target_uniprot (str): The Uniprot ID of the target protein.
        cell_line (str): The cell line identifier.
        device (str): The device to run the model on.
        proba_threshold (float): The probability threshold.
        use_majority_vote (bool): Whether to use the majority vote.
        use_models_from_cv (bool): Whether to use the models from cross-validation.
        use_xgboost_models (bool): Whether to use the XGBoost models.
        study_type (str): Use models trained on the specified study. Options are 'standard', 'similarity', 'target'.

    Returns:
        bool: Whether the PROTAC is active or not.
    """
    pred = get_protac_active_proba(
        protac_smiles,
        e3_ligase,
        target_uniprot,
        cell_line,
        device,
        use_models_from_cv,
        use_xgboost_models,
        study_type,
    )
    if use_majority_vote:
        return pred['majority_vote']
    else:
        return pred['mean'] > proba_threshold


def get_protac_embedding(
        protac_smiles: str | List[str],
        e3_ligase: str | List[str],
        target_uniprot: str | List[str],
        cell_line: str | List[str],
        device: Literal['cpu', 'cuda'] = 'cpu',
        use_models_from_cv: bool = False,
        study_type: Literal['standard', 'similarity', 'target'] = 'standard',
) -> Dict[str, np.ndarray]:
    """ Get the embeddings of a PROTAC or a list of PROTACs.

    Args:
        protac_smiles (str | List[str]): The SMILES of the PROTAC.
        e3_ligase (str | List[str]): The Uniprot ID of the E3 ligase.
        target_uniprot (str | List[str]): The Uniprot ID of the target protein.
        cell_line (str | List[str]): The cell line identifier.
        device (str): The device to run the model on.
        use_models_from_cv (bool): Whether to use the models from cross-validation.
        study_type (str): Use models trained on the specified study. Options are 'standard', 'similarity', 'target'.

    Returns:
        Dict[str, np.ndarray]: The embeddings of the given PROTAC. Each key is the name of the model and the value is the embedding, of shape: (batch_size, model_hidden_size). NOTE: Each model has its own hidden size, so the embeddings might have different dimensions.
    """
    # Check that the study type is valid
    if study_type not in ['standard', 'similarity', 'target']:
        raise ValueError(f"Invalid study type: {study_type}. Options are 'standard', 'similarity', 'target'.")

    # Check that the device is valid
    if device not in ['cpu', 'cuda']:
        raise ValueError(f"Invalid device: {device}. Options are 'cpu', 'cuda'.")
    
    # Check that if any the models input is a list, all inputs are lists
    model_inputs = [protac_smiles, e3_ligase, target_uniprot, cell_line]
    if any(isinstance(i, list) for i in model_inputs):
        if not all(isinstance(i, list) for i in model_inputs):
            raise ValueError("All model inputs must be lists if one of the inputs is a list.")

    # Load all required models in pkg_resources
    device = torch.device(device)
    models = {}
    model_to_load = 'best_model' if not use_models_from_cv else 'cv_model'
    for model_filename in pkg_resources.resource_listdir(__name__, 'models'):
        if model_to_load not in model_filename:
            continue
        if study_type not in model_filename:
            continue
        if 'xgboost' not in model_filename:
            ckpt_path = pkg_resources.resource_filename(__name__, f'models/{model_filename}')
            models[ckpt_path] = load_model(ckpt_path).to(device)

    protein2embedding = load_protein2embedding()
    cell2embedding = load_cell2embedding()

    # Get the dimension of the embeddings from the first np.array in the dictionary
    protein_embedding_size = next(iter(protein2embedding.values())).shape[0]
    cell_embedding_size = next(iter(cell2embedding.values())).shape[0]
    # Setup default embeddings
    default_protein_emb = np.zeros(protein_embedding_size)
    default_cell_emb = np.zeros(cell_embedding_size)

    # Check if any model name contains cellsonehot, if so, get onehot encoding
    cell2onehot = None
    if any('cellsonehot' in model_name for model_name in models.keys()):
        onehotenc = OneHotEncoder(sparse_output=False)
        cell_embeddings = onehotenc.fit_transform(
            np.array(list(cell2embedding.keys())).reshape(-1, 1)
        )
        cell2onehot = {k: v for k, v in zip(cell2embedding.keys(), cell_embeddings)}
    
    # Check if any of the model names contain aminoacidcnt, if so, get the CountVectorizer
    protein2aacnt = None
    if any('aminoacidcnt' in model_name for model_name in models.keys()):
        # Create a new protein2embedding dictionary with amino acid sequence
        protac_df = load_curated_dataset()
        # Create the dictionary mapping 'Uniprot' to 'POI Sequence'
        protein2aacnt = protac_df.set_index('Uniprot')['POI Sequence'].to_dict()
        # Create the dictionary mapping 'E3 Ligase Uniprot' to 'E3 Ligase Sequence'
        e32seq = protac_df.set_index('E3 Ligase Uniprot')['E3 Ligase Sequence'].to_dict()
        # Merge the two dictionaries into a new protein2aacnt dictionary
        protein2aacnt.update(e32seq)

        # Get count vectorized embeddings for proteins
        # NOTE: Check that the protein2aacnt is a dictionary of strings
        if not all(isinstance(k, str) for k in protein2aacnt.keys()):
            raise ValueError("All keys in `protein2aacnt` must be strings.")
        countvec = CountVectorizer(ngram_range=(1, 1), analyzer='char')
        protein_embeddings = countvec.fit_transform(
            list(protein2aacnt.keys())
        ).toarray()
        protein2aacnt = {k: v for k, v in zip(protein2aacnt.keys(), protein_embeddings)}

    # Convert the E3 ligase to Uniprot ID
    if isinstance(e3_ligase, list):
        e3_ligase_uniprot = [config.e3_ligase2uniprot.get(e3, '') for e3 in e3_ligase]
    else:
        e3_ligase_uniprot = config.e3_ligase2uniprot.get(e3_ligase, '')

    # Get the embeddings for the PROTAC, E3 ligase, target protein, and cell line
    # Check if the input is a list or a single string, in the latter case,
    # convert to a list to create a batch of size 1, len(list) otherwise.
    if isinstance(protac_smiles, list):
        # TODO: Add warning on missing entries?
        smiles_emb = [get_fingerprint(s) for s in protac_smiles]
        cell_emb = [cell2embedding.get(c, default_cell_emb) for c in cell_line]
        e3_emb = [protein2embedding.get(e3, default_protein_emb) for e3 in e3_ligase_uniprot]
        poi_emb = [protein2embedding.get(t, default_protein_emb) for t in target_uniprot]
        # Convert to one-hot encoded cell embeddings if necessary
        if cell2onehot is not None:
            cell_onehot = [cell2onehot.get(c, default_cell_emb) for c in cell_line]
        # Convert to amino acid count embeddings if necessary
        if protein2aacnt is not None:
            poi_aacnt = [protein2aacnt.get(t, default_protein_emb) for t in target_uniprot]
            e3_aacnt = [protein2aacnt.get(e3, default_protein_emb) for e3 in e3_ligase_uniprot]
    else:
        if e3_ligase not in config.e3_ligase2uniprot:
            available_e3_ligases = ', '.join(list(config.e3_ligase2uniprot.keys()))
            logging.warning(f"The E3 ligase {e3_ligase} is not in the database. Using the default E3 ligase. Available E3 ligases are: {available_e3_ligases}")
        if target_uniprot not in protein2embedding:
            logging.warning(f"The target protein {target_uniprot} is not in the database. Using the default target protein.")
        if cell_line not in cell2embedding:
            logging.warning(f"The cell line {cell_line} is not in the database. Using the default cell line.")
        smiles_emb = [get_fingerprint(protac_smiles)]
        cell_emb = [cell2embedding.get(cell_line, default_cell_emb)]
        poi_emb = [protein2embedding.get(target_uniprot, default_protein_emb)]
        e3_emb = [protein2embedding.get(e3_ligase_uniprot, default_protein_emb)]
        # Convert to one-hot encoded cell embeddings if necessary
        if cell2onehot is not None:
            cell_onehot = [cell2onehot.get(cell_line, default_cell_emb)]
        # Convert to amino acid count embeddings if necessary
        if protein2aacnt is not None:
            poi_aacnt = [protein2aacnt.get(target_uniprot, default_protein_emb)]
            e3_aacnt = [protein2aacnt.get(e3_ligase_uniprot, default_protein_emb)]

    # Convert to numpy arrays
    smiles_emb = np.array(smiles_emb)
    cell_emb = np.array(cell_emb)
    poi_emb = np.array(poi_emb)
    e3_emb = np.array(e3_emb)
    if cell2onehot is not None:
        cell_onehot = np.array(cell_onehot)
    if protein2aacnt is not None:
        poi_aacnt = np.array(poi_aacnt)
        e3_aacnt = np.array(e3_aacnt)

    # Convert to torch tensors
    smiles_emb = torch.tensor(smiles_emb).float().to(device)
    cell_emb = torch.tensor(cell_emb).to(device)
    poi_emb = torch.tensor(poi_emb).to(device)
    e3_emb = torch.tensor(e3_emb).to(device)
    if cell2onehot is not None:
        cell_onehot = torch.tensor(cell_onehot).float().to(device)
    if protein2aacnt is not None:
        poi_aacnt = torch.tensor(poi_aacnt).float().to(device)
        e3_aacnt = torch.tensor(e3_aacnt).float().to(device)
    
    # Average the predictions of all models
    protac_embs = {}
    for ckpt_path, model in models.items():
        # Get the last part of the path
        ckpt_path = os.path.basename(ckpt_path)
        _, protac_emb = model(
            poi_emb if 'aminoacidcnt' not in ckpt_path else poi_aacnt,
            e3_emb if 'aminoacidcnt' not in ckpt_path else e3_aacnt,
            cell_emb if 'cellsonehot' not in ckpt_path else cell_onehot,
            smiles_emb,
            prescaled_embeddings=False, # Normalization performed by the model
            return_embeddings=True,
        )
        protac_embs[ckpt_path] = protac_emb
    
    return protac_embs