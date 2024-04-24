import pytest
import os
import sys
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from protac_degradation_predictor import PROTAC_Model, PROTAC_Predictor

import torch


def test_protac_model():
    model = PROTAC_Model(hidden_dim=128)
    assert model.hidden_dim == 128
    assert model.smiles_emb_dim == 224
    assert model.poi_emb_dim == 1024
    assert model.e3_emb_dim == 1024
    assert model.cell_emb_dim == 768
    assert model.batch_size == 32
    assert model.learning_rate == 0.001
    assert model.dropout == 0.2
    assert model.join_embeddings == 'concat'
    assert model.train_dataset is None
    assert model.val_dataset is None
    assert model.test_dataset is None
    assert model.disabled_embeddings == []
    assert model.apply_scaling == False

def test_protac_predictor():
    predictor = PROTAC_Predictor(hidden_dim=128)
    assert predictor.hidden_dim == 128
    assert predictor.smiles_emb_dim == 224
    assert predictor.poi_emb_dim == 1024
    assert predictor.e3_emb_dim == 1024
    assert predictor.cell_emb_dim == 768
    assert predictor.join_embeddings == 'concat'
    assert predictor.disabled_embeddings == []

def test_load_model(caplog):
    caplog.set_level(logging.WARNING)

    model = PROTAC_Model.load_from_checkpoint(
        'data/test_model.ckpt',
        map_location=torch.device("cpu") if not torch.cuda.is_available() else None,
    )
    # apply_scaling: true
    # batch_size: 8
    # cell_emb_dim: 768
    # disabled_embeddings: []
    # dropout: 0.1498104322091649
    # e3_emb_dim: 1024
    # hidden_dim: 768
    # join_embeddings: concat
    # learning_rate: 4.881387978425994e-05
    # poi_emb_dim: 1024
    # smiles_emb_dim: 224
    assert model.hidden_dim == 768
    assert model.smiles_emb_dim == 224
    assert model.poi_emb_dim == 1024
    assert model.e3_emb_dim == 1024
    assert model.cell_emb_dim == 768
    assert model.batch_size == 8
    assert model.learning_rate == 4.881387978425994e-05
    assert model.dropout == 0.1498104322091649
    assert model.join_embeddings == 'concat'
    assert model.disabled_embeddings == []
    assert model.apply_scaling == True


def test_checkpoint_file():
    checkpoint = torch.load(
        'data/test_model.ckpt',
        map_location=torch.device("cpu") if not torch.cuda.is_available() else None,
    )
    print(checkpoint.keys())
    print(checkpoint["hyper_parameters"])
    print([k for k, v in checkpoint["state_dict"].items()])

pytest.main()
