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
    assert model.smiles_emb_dim == 256
    assert model.poi_emb_dim == 1024
    assert model.e3_emb_dim == 1024
    assert model.cell_emb_dim == 768
    assert model.batch_size == 128
    assert model.learning_rate == 0.001
    assert model.dropout == 0.2
    assert model.join_embeddings == 'sum'
    assert model.train_dataset is None
    assert model.val_dataset is None
    assert model.test_dataset is None
    assert model.disabled_embeddings == []
    assert model.apply_scaling == True

def test_protac_predictor():
    predictor = PROTAC_Predictor(hidden_dim=128)
    assert predictor.hidden_dim == 128
    assert predictor.smiles_emb_dim == 256
    assert predictor.poi_emb_dim == 1024
    assert predictor.e3_emb_dim == 1024
    assert predictor.cell_emb_dim == 768
    assert predictor.join_embeddings == 'sum'
    assert predictor.disabled_embeddings == []

def test_load_model(caplog):
    # caplog.set_level(logging.WARNING)

    # model_filename = 'data/best_model_n0_random-epoch=6-val_acc=0.74-val_roc_auc=0.796.ckpt'

    # model = PROTAC_Model.load_from_checkpoint(
    #     model_filename,
    #     map_location=torch.device("cpu") if not torch.cuda.is_available() else None,
    # )
    # assert model.hidden_dim == 768
    # assert model.smiles_emb_dim == 224
    # assert model.poi_emb_dim == 1024
    # assert model.e3_emb_dim == 1024
    # assert model.cell_emb_dim == 768
    # assert model.batch_size == 8
    # assert model.learning_rate == 1.843233973932415e-05
    # assert model.dropout == 0.11257777663560328
    # assert model.join_embeddings == 'concat'
    # assert model.disabled_embeddings == []
    # assert model.apply_scaling == True
    # print(model.scalers)
    pass


def test_checkpoint_file():
    # model_filename = 'data/best_model_n0_random-epoch=6-val_acc=0.74-val_roc_auc=0.796.ckpt'
    # checkpoint = torch.load(
    #     model_filename,
    #     map_location=torch.device("cpu") if not torch.cuda.is_available() else None,
    # )
    # print(checkpoint.keys())
    # print(checkpoint["hyper_parameters"])
    # print([k for k, v in checkpoint["state_dict"].items()])
    # import pickle

    # print(pickle.loads(checkpoint['scalers']))
    pass

pytest.main()
