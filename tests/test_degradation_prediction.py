import pytest
import os
import sys
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from protac_degradation_predictor import (
    get_protac_active_proba,
    is_protac_active,
)

import torch


def test_active_proba():
    protac_smiles = 'Cc1ncsc1-c1ccc([C@H](C)NC(=O)[C@@H]2C[C@@H](O)CN2C(=O)[C@@H](NC(=O)CC(=O)N2CCN(CC[C@H](CSc3ccccc3)Nc3ccc(S(=O)(=O)NC(=O)c4ccc(N5CCN(CC6=C(c7ccc(Cl)cc7)CCC(C)(C)C6)CC5)cc4)cc3S(=O)(=O)C(F)(F)F)CC2)C(C)(C)C)cc1'
    e3_ligase = 'VHL'
    target_uniprot = 'Q07817'
    cell_line = 'MOLT-4'
    device = 'cpu'

    active_prob = get_protac_active_proba(
        protac_smiles=protac_smiles,
        e3_ligase=e3_ligase,
        target_uniprot=target_uniprot,
        cell_line=cell_line,
        device=device,
    )

    print(f'Active probability: {active_prob} (CPU)')

    if torch.cuda.is_available():
        active_prob = get_protac_active_proba(
            protac_smiles=[protac_smiles] * 16,
            e3_ligase=[e3_ligase] * 16,
            target_uniprot=[target_uniprot] * 16,
            cell_line=[cell_line] * 16,
            device='cuda' if torch.cuda.is_available() else 'cpu',
        )

        print(f'Active probability: {active_prob} (GPU)')


def test_xgboost_active_proba():
    protac_smiles = 'Cc1ncsc1-c1ccc([C@H](C)NC(=O)[C@@H]2C[C@@H](O)CN2C(=O)[C@@H](NC(=O)CC(=O)N2CCN(CC[C@H](CSc3ccccc3)Nc3ccc(S(=O)(=O)NC(=O)c4ccc(N5CCN(CC6=C(c7ccc(Cl)cc7)CCC(C)(C)C6)CC5)cc4)cc3S(=O)(=O)C(F)(F)F)CC2)C(C)(C)C)cc1'
    e3_ligase = 'VHL'
    target_uniprot = 'Q07817'
    cell_line = 'MOLT-4'
    device = 'cpu'

    active_prob = get_protac_active_proba(
        protac_smiles=protac_smiles,
        e3_ligase=e3_ligase,
        target_uniprot=target_uniprot,
        cell_line=cell_line,
        device=device,
        use_xgboost_models=True,
    )

    print(f'Active probability: {active_prob} (CPU)')    


def test_active_proba_list():
    protac_smiles = 'Cc1ncsc1-c1ccc([C@H](C)NC(=O)[C@@H]2C[C@@H](O)CN2C(=O)[C@@H](NC(=O)CC(=O)N2CCN(CC[C@H](CSc3ccccc3)Nc3ccc(S(=O)(=O)NC(=O)c4ccc(N5CCN(CC6=C(c7ccc(Cl)cc7)CCC(C)(C)C6)CC5)cc4)cc3S(=O)(=O)C(F)(F)F)CC2)C(C)(C)C)cc1'
    e3_ligase = 'VHL'
    target_uniprot = 'Q07817'
    cell_line = 'MOLT-4'
    device = 'cpu'
    batch_size = 32

    active_prob = get_protac_active_proba(
        protac_smiles=[protac_smiles] * batch_size,
        e3_ligase=[e3_ligase] * batch_size,
        target_uniprot=[target_uniprot] * batch_size,
        cell_line=[cell_line] * batch_size,
        device=device,
    )

def test_xgboost_active_proba_list():
    protac_smiles = 'Cc1ncsc1-c1ccc([C@H](C)NC(=O)[C@@H]2C[C@@H](O)CN2C(=O)[C@@H](NC(=O)CC(=O)N2CCN(CC[C@H](CSc3ccccc3)Nc3ccc(S(=O)(=O)NC(=O)c4ccc(N5CCN(CC6=C(c7ccc(Cl)cc7)CCC(C)(C)C6)CC5)cc4)cc3S(=O)(=O)C(F)(F)F)CC2)C(C)(C)C)cc1'
    e3_ligase = 'VHL'
    target_uniprot = 'Q07817'
    cell_line = 'MOLT-4'
    device = 'cpu'
    batch_size = 32

    active_prob = get_protac_active_proba(
        protac_smiles=[protac_smiles] * batch_size,
        e3_ligase=[e3_ligase] * batch_size,
        target_uniprot=[target_uniprot] * batch_size,
        cell_line=[cell_line] * batch_size,
        device=device,
        use_xgboost_models=True,
    )

def test_is_protac_active():
    pass