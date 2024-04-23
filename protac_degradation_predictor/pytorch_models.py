import warnings
from typing import Literal, List, Tuple, Optional, Dict

from .protac_dataset import PROTAC_Dataset
from .config import Config

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchmetrics import (
    Accuracy,
    AUROC,
    Precision,
    Recall,
    F1Score,
    MetricCollection,
)
from imblearn.over_sampling import SMOTE


class PROTAC_Predictor(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        smiles_emb_dim: int = Config.fingerprint_size,
        poi_emb_dim: int = Config.protein_embedding_size,
        e3_emb_dim: int = Config.protein_embedding_size,
        cell_emb_dim: int = Config.cell_embedding_size,
        dropout: float = 0.2,
        join_embeddings: Literal['beginning', 'concat', 'sum'] = 'concat',
        disabled_embeddings: list = [],
    ):
        """ Initialize the PROTAC model.
        
        Args:
            hidden_dim (int): The hidden dimension of the model
            smiles_emb_dim (int): The dimension of the SMILES embeddings
            poi_emb_dim (int): The dimension of the POI embeddings
            e3_emb_dim (int): The dimension of the E3 Ligase embeddings
            cell_emb_dim (int): The dimension of the cell line embeddings
            dropout (float): The dropout rate
            join_embeddings (Literal['beginning', 'concat', 'sum']): How to join the embeddings
            disabled_embeddings (list): List of disabled embeddings. Can be 'poi', 'e3', 'cell', 'smiles'
        """
        super().__init__()
        self.poi_emb_dim = poi_emb_dim
        self.e3_emb_dim = e3_emb_dim
        self.cell_emb_dim = cell_emb_dim
        self.smiles_emb_dim = smiles_emb_dim
        self.hidden_dim = hidden_dim
        self.join_embeddings = join_embeddings
        self.disabled_embeddings = disabled_embeddings
        # Set our init args as class attributes
        self.__dict__.update(locals())

        # Define "surrogate models" branches
        if self.join_embeddings != 'beginning':
            if 'poi' not in self.disabled_embeddings:
                self.poi_emb = nn.Linear(poi_emb_dim, hidden_dim)
            if 'e3' not in self.disabled_embeddings:
                self.e3_emb = nn.Linear(e3_emb_dim, hidden_dim)
            if 'cell' not in self.disabled_embeddings:
                self.cell_emb = nn.Linear(cell_emb_dim, hidden_dim)
            if 'smiles' not in self.disabled_embeddings:
                self.smiles_emb = nn.Linear(smiles_emb_dim, hidden_dim)

        # Define hidden dimension for joining layer
        if self.join_embeddings == 'beginning':
            joint_dim = smiles_emb_dim if 'smiles' not in self.disabled_embeddings else 0
            joint_dim += poi_emb_dim if 'poi' not in self.disabled_embeddings else 0
            joint_dim += e3_emb_dim if 'e3' not in self.disabled_embeddings else 0
            joint_dim += cell_emb_dim if 'cell' not in self.disabled_embeddings else 0
        elif self.join_embeddings == 'concat':
            joint_dim = hidden_dim * (4 - len(self.disabled_embeddings))
        elif self.join_embeddings == 'sum':
            joint_dim = hidden_dim

        self.fc0 = nn.Linear(joint_dim, joint_dim)
        self.fc1 = nn.Linear(joint_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        self.dropout = nn.Dropout(p=dropout)

    
    def forward(self, poi_emb, e3_emb, cell_emb, smiles_emb):
        embeddings = []
        if self.join_embeddings == 'beginning':
            if 'poi' not in self.disabled_embeddings:
                embeddings.append(poi_emb)
            if 'e3' not in self.disabled_embeddings:
                embeddings.append(e3_emb)
            if 'cell' not in self.disabled_embeddings:
                embeddings.append(cell_emb)
            if 'smiles' not in self.disabled_embeddings:
                embeddings.append(smiles_emb)
            x = torch.cat(embeddings, dim=1)
            x = self.dropout(F.relu(self.fc0(x)))
        else:
            if 'poi' not in self.disabled_embeddings:
                embeddings.append(self.poi_emb(poi_emb))
            if 'e3' not in self.disabled_embeddings:
                embeddings.append(self.e3_emb(e3_emb))
            if 'cell' not in self.disabled_embeddings:
                embeddings.append(self.cell_emb(cell_emb))
            if 'smiles' not in self.disabled_embeddings:
                embeddings.append(self.smiles_emb(smiles_emb))
            if self.join_embeddings == 'concat':
                x = torch.cat(embeddings, dim=1)
            elif self.join_embeddings == 'sum':
                if len(embeddings) > 1:
                    embeddings = torch.stack(embeddings, dim=1)
                    x = torch.sum(embeddings, dim=1)
                else:
                    x = embeddings[0]
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x



class PROTAC_Model(pl.LightningModule):

    def __init__(
        self,
        hidden_dim: int,
        smiles_emb_dim: int = 224,
        poi_emb_dim: int = 1024,
        e3_emb_dim: int = 1024,
        cell_emb_dim: int = 768,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        dropout: float = 0.2,
        join_embeddings: Literal['beginning', 'concat', 'sum'] = 'concat',
        train_dataset: PROTAC_Dataset = None,
        val_dataset: PROTAC_Dataset = None,
        test_dataset: PROTAC_Dataset = None,
        disabled_embeddings: list = [],
        apply_scaling: bool = False,
    ):
        """ Initialize the PROTAC Pytorch Lightning model.
        
        Args:
            hidden_dim (int): The hidden dimension of the model
            smiles_emb_dim (int): The dimension of the SMILES embeddings
            poi_emb_dim (int): The dimension of the POI embeddings
            e3_emb_dim (int): The dimension of the E3 Ligase embeddings
            cell_emb_dim (int): The dimension of the cell line embeddings
            batch_size (int): The batch size
            learning_rate (float): The learning rate
            dropout (float): The dropout rate
            join_embeddings (Literal['beginning', 'concat', 'sum']): How to join the embeddings
            train_dataset (PROTAC_Dataset): The training dataset
            val_dataset (PROTAC_Dataset): The validation dataset
            test_dataset (PROTAC_Dataset): The test dataset
            disabled_embeddings (list): List of disabled embeddings. Can be 'poi', 'e3', 'cell', 'smiles'
            apply_scaling (bool): Whether to apply scaling to the embeddings
        """
        super().__init__()
        self.poi_emb_dim = poi_emb_dim
        self.e3_emb_dim = e3_emb_dim
        self.cell_emb_dim = cell_emb_dim
        self.smiles_emb_dim = smiles_emb_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.join_embeddings = join_embeddings
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.disabled_embeddings = disabled_embeddings
        self.apply_scaling = apply_scaling
        # Set our init args as class attributes
        self.__dict__.update(locals())  # Add arguments as attributes
        # Save the arguments passed to init
        ignore_args_as_hyperparams = [
            'train_dataset',
            'test_dataset',
            'val_dataset',
        ]
        self.save_hyperparameters(ignore=ignore_args_as_hyperparams)

        self.model = PROTAC_Predictor(
            hidden_dim=hidden_dim,
            smiles_emb_dim=smiles_emb_dim,
            poi_emb_dim=poi_emb_dim,
            e3_emb_dim=e3_emb_dim,
            cell_emb_dim=cell_emb_dim,
            dropout=dropout,
            join_embeddings=join_embeddings,
            disabled_embeddings=disabled_embeddings,
        )

        stages = ['train_metrics', 'val_metrics', 'test_metrics']
        self.metrics = nn.ModuleDict({s: MetricCollection({
            'acc': Accuracy(task='binary'),
            'roc_auc': AUROC(task='binary'),
            'precision': Precision(task='binary'),
            'recall': Recall(task='binary'),
            'f1_score': F1Score(task='binary'),
            'opt_score': Accuracy(task='binary') + F1Score(task='binary'),
            'hp_metric': Accuracy(task='binary'),
        }, prefix=s.replace('metrics', '')) for s in stages})

        # Misc settings
        self.missing_dataset_error = \
            '''Class variable `{0}` is None. If the model was loaded from a checkpoint, the dataset must be set manually:
            
            model = {1}.load_from_checkpoint('checkpoint.ckpt')
            model.{0} = my_{0}
            '''
        
        # Apply scaling in datasets
        if self.apply_scaling:
            use_single_scaler = True if self.join_embeddings == 'beginning' else False
            self.scalers = self.train_dataset.fit_scaling(use_single_scaler)
            self.train_dataset.apply_scaling(self.scalers, use_single_scaler)
            self.val_dataset.apply_scaling(self.scalers, use_single_scaler)
            if self.test_dataset:
                self.test_dataset.apply_scaling(self.scalers, use_single_scaler)

    def forward(self, poi_emb, e3_emb, cell_emb, smiles_emb):
        return self.model(poi_emb, e3_emb, cell_emb, smiles_emb)

    def step(self, batch, batch_idx, stage):
        poi_emb = batch['poi_emb']
        e3_emb = batch['e3_emb']
        cell_emb = batch['cell_emb']
        smiles_emb = batch['smiles_emb']
        y = batch['active'].float().unsqueeze(1)

        y_hat = self.forward(poi_emb, e3_emb, cell_emb, smiles_emb)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)

        self.metrics[f'{stage}_metrics'].update(y_hat, y)
        self.log(f'{stage}_loss', loss, on_epoch=True, prog_bar=True)
        self.log_dict(self.metrics[f'{stage}_metrics'], on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'test')

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    def predict_step(self, batch, batch_idx):
        poi_emb = batch['poi_emb']
        e3_emb = batch['e3_emb']
        cell_emb = batch['cell_emb']
        smiles_emb = batch['smiles_emb']

        if self.apply_scaling:
            if self.join_embeddings == 'beginning':
                embeddings = np.hstack([
                    np.array(smiles_emb.tolist()),
                    np.array(poi_emb.tolist()),
                    np.array(e3_emb.tolist()),
                    np.array(cell_emb.tolist()),
                ])
                embeddings = self.scalers.transform(embeddings)
                smiles_emb = embeddings[:, :self.smiles_emb_dim]
                poi_emb = embeddings[:, self.smiles_emb_dim:self.smiles_emb_dim+self.poi_emb_dim]
                e3_emb = embeddings[:, self.smiles_emb_dim+self.poi_emb_dim:self.smiles_emb_dim+2*self.poi_emb_dim]
                cell_emb = embeddings[:, -self.cell_emb_dim:]
            else:
                poi_emb = self.scalers['Uniprot'].transform(poi_emb)
                e3_emb = self.scalers['E3 Ligase Uniprot'].transform(e3_emb)
                cell_emb = self.scalers['Cell Line Identifier'].transform(cell_emb)
                smiles_emb = self.scalers['Smiles'].transform(smiles_emb)

        y_hat = self.forward(poi_emb, e3_emb, cell_emb, smiles_emb)
        return torch.sigmoid(y_hat)

    def train_dataloader(self):
        if self.train_dataset is None:
            format = 'train_dataset', self.__class__.__name__
            raise ValueError(self.missing_dataset_error.format(*format))
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            # drop_last=True,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            format = 'val_dataset', self.__class__.__name__
            raise ValueError(self.missing_dataset_error.format(*format))
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            format = 'test_dataset', self.__class__.__name__
            raise ValueError(self.missing_dataset_error.format(*format))
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )


def train_model(
        protein2embedding: Dict,
        cell2embedding: Dict,
        smiles2fp: Dict,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: Optional[pd.DataFrame] = None,
        hidden_dim: int = 768,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        dropout: float = 0.2,
        max_epochs: int = 50,
        smiles_emb_dim: int = 224,
        join_embeddings: Literal['beginning', 'concat', 'sum'] = 'concat',
        smote_k_neighbors:int = 5,
        use_smote: bool = True,
        apply_scaling: bool = False,
        active_label: str = 'Active',
        fast_dev_run: bool = False,
        use_logger: bool = True,
        logger_name: str = 'protac',
        disabled_embeddings: List[str] = [],
) -> tuple:
    """ Train a PROTAC model using the given datasets and hyperparameters.
    
    Args:
        protein2embedding (dict): Dictionary of protein embeddings.
        cell2embedding (dict): Dictionary of cell line embeddings.
        smiles2fp (dict): Dictionary of SMILES to fingerprint.
        train_df (pd.DataFrame): The training set. It must include the following columns: 'Smiles', 'Uniprot', 'E3 Ligase Uniprot', 'Cell Line Identifier', <active_label>.
        val_df (pd.DataFrame): The validation set.  It must include the following columns: 'Smiles', 'Uniprot', 'E3 Ligase Uniprot', 'Cell Line Identifier', <active_label>.
        test_df (pd.DataFrame): The test set. If provided, the returned metrics will include test performance.  It must include the following columns: 'Smiles', 'Uniprot', 'E3 Ligase Uniprot', 'Cell Line Identifier', <active_label>.
        hidden_dim (int): The hidden dimension of the model.
        batch_size (int): The batch size.
        learning_rate (float): The learning rate.
        max_epochs (int): The maximum number of epochs.
        smiles_emb_dim (int): The dimension of the SMILES embeddings.
        smote_k_neighbors (int): The number of neighbors for the SMOTE oversampler.
        fast_dev_run (bool): Whether to run a fast development run.
        disabled_embeddings (list): The list of disabled embeddings.
    
    Returns:
        tuple: The trained model, the trainer, and the metrics.
    """
    oversampler = SMOTE(k_neighbors=smote_k_neighbors, random_state=42)
    train_ds = PROTAC_Dataset(
        train_df,
        protein2embedding,
        cell2embedding,
        smiles2fp,
        use_smote=use_smote,
        oversampler=oversampler if use_smote else None,
        active_label=active_label,
    )
    val_ds = PROTAC_Dataset(
        val_df,
        protein2embedding,
        cell2embedding,
        smiles2fp,
        active_label=active_label,
    )
    if test_df is not None:
        test_ds = PROTAC_Dataset(
            test_df,
            protein2embedding,
            cell2embedding,
            smiles2fp,
            active_label=active_label,
        )
    logger = pl.loggers.TensorBoardLogger(
        save_dir='../logs',
        name=logger_name,
    )
    callbacks = [
        pl.callbacks.EarlyStopping(
            monitor='train_loss',
            patience=10,
            mode='min',
            verbose=False,
        ),
        pl.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            mode='min',
            verbose=False,
        ),
        pl.callbacks.EarlyStopping(
            monitor='val_acc',
            patience=10,
            mode='max',
            verbose=False,
        ),
        # pl.callbacks.ModelCheckpoint(
        #     monitor='val_acc',
        #     mode='max',
        #     verbose=True,
        #     filename='{epoch}-{val_metrics_opt_score:.4f}',
        # ),
    ]
    # Define Trainer
    trainer = pl.Trainer(
        logger=logger if use_logger else False,
        callbacks=callbacks,
        max_epochs=max_epochs,
        fast_dev_run=fast_dev_run,
        enable_model_summary=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        devices=1,
        num_nodes=1,
    )
    model = PROTAC_Model(
        hidden_dim=hidden_dim,
        smiles_emb_dim=smiles_emb_dim,
        poi_emb_dim=1024,
        e3_emb_dim=1024,
        cell_emb_dim=768,
        batch_size=batch_size,
        join_embeddings=join_embeddings,
        dropout=dropout,
        learning_rate=learning_rate,
        apply_scaling=apply_scaling,
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds if test_df is not None else None,
        disabled_embeddings=disabled_embeddings,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trainer.fit(model)
    metrics = trainer.validate(model, verbose=False)[0]
    if test_df is not None:
        test_metrics = trainer.test(model, verbose=False)[0]
        metrics.update(test_metrics)
    return model, trainer, metrics


def load_model(
        ckpt_path: str,
) -> PROTAC_Model:
    """ Load a PROTAC model from a checkpoint.
    
    Args:
        ckpt_path (str): The path to the checkpoint.
    
    Returns:
        PROTAC_Model: The loaded model.
    """
    model = PROTAC_Model.load_from_checkpoint(ckpt_path)
    model.eval()
    return model