import warnings
import pickle
import logging
from typing import Literal, List, Tuple, Optional, Dict

from .protac_dataset import PROTAC_Dataset, get_datasets
from .config import config

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
from sklearn.preprocessing import StandardScaler


class PROTAC_Predictor(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        smiles_emb_dim: int = config.fingerprint_size,
        poi_emb_dim: int = config.protein_embedding_size,
        e3_emb_dim: int = config.protein_embedding_size,
        cell_emb_dim: int = config.cell_embedding_size,
        dropout: float = 0.2,
        join_embeddings: Literal['beginning', 'concat', 'sum'] = 'sum',
        use_batch_norm: bool = False,
        disabled_embeddings: List[Literal['smiles', 'poi', 'e3', 'cell']] = [],
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
        # Set our init args as class attributes
        self.__dict__.update(locals())

        # Define "surrogate models" branches
        # NOTE: The softmax is used to ensure that the embeddings are normalized
        # and can be summed on a "similar scale".
        if self.join_embeddings != 'beginning':
            if 'poi' not in self.disabled_embeddings:
                self.poi_fc = nn.Sequential(
                    nn.Linear(poi_emb_dim, hidden_dim),
                    nn.Softmax(dim=1),
                )
            if 'e3' not in self.disabled_embeddings:
                self.e3_fc = nn.Sequential(
                    nn.Linear(e3_emb_dim, hidden_dim),
                    nn.Softmax(dim=1),
                )
            if 'cell' not in self.disabled_embeddings:
                self.cell_fc = nn.Sequential(
                    nn.Linear(cell_emb_dim, hidden_dim),
                    nn.Softmax(dim=1),
                )
            if 'smiles' not in self.disabled_embeddings:
                self.smiles_emb = nn.Sequential(
                    nn.Linear(smiles_emb_dim, hidden_dim),
                    nn.Softmax(dim=1),
                )

        # Define hidden dimension for joining layer
        if self.join_embeddings == 'beginning':
            joint_dim = smiles_emb_dim if 'smiles' not in self.disabled_embeddings else 0
            joint_dim += poi_emb_dim if 'poi' not in self.disabled_embeddings else 0
            joint_dim += e3_emb_dim if 'e3' not in self.disabled_embeddings else 0
            joint_dim += cell_emb_dim if 'cell' not in self.disabled_embeddings else 0
            self.fc0 = nn.Linear(joint_dim, joint_dim)
        elif self.join_embeddings == 'concat':
            joint_dim = hidden_dim * (4 - len(self.disabled_embeddings))
        elif self.join_embeddings == 'sum':
            joint_dim = hidden_dim

        self.fc1 = nn.Linear(joint_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        self.bnorm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(p=dropout)

    
    def forward(self, poi_emb, e3_emb, cell_emb, smiles_emb, return_embeddings=False):
        embeddings = []
        if self.join_embeddings == 'beginning':
            # TODO: Remove this if-branch
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
                embeddings.append(self.poi_fc(poi_emb))
                if torch.isnan(embeddings[-1]).any():
                    raise ValueError("NaN values found in POI embeddings.")
                
            if 'e3' not in self.disabled_embeddings:
                embeddings.append(self.e3_fc(e3_emb))
                if torch.isnan(embeddings[-1]).any():
                    raise ValueError("NaN values found in E3 embeddings.")
            
            if 'cell' not in self.disabled_embeddings:
                embeddings.append(self.cell_fc(cell_emb))
                if torch.isnan(embeddings[-1]).any():
                    raise ValueError("NaN values found in cell embeddings.")
                
            if 'smiles' not in self.disabled_embeddings:
                embeddings.append(self.smiles_emb(smiles_emb))
                if torch.isnan(embeddings[-1]).any():
                    raise ValueError("NaN values found in SMILES embeddings.")
                
            if self.join_embeddings == 'concat':
                x = torch.cat(embeddings, dim=1)
            elif self.join_embeddings == 'sum':
                if len(embeddings) > 1:
                    embeddings = torch.stack(embeddings, dim=1)
                    x = torch.sum(embeddings, dim=1)
                else:
                    x = embeddings[0]
        if torch.isnan(x).any():
            raise ValueError("NaN values found in sum of softmax-ed embeddings.")
        x = F.relu(self.fc1(x))
        h = self.bnorm(x) if self.use_batch_norm else self.self.dropout(x)
        x = self.fc3(h)
        if return_embeddings:
            return x, h
        return x


class PROTAC_Model(pl.LightningModule):

    def __init__(
        self,
        hidden_dim: int,
        smiles_emb_dim: int = config.fingerprint_size,
        poi_emb_dim: int = config.protein_embedding_size,
        e3_emb_dim: int = config.protein_embedding_size,
        cell_emb_dim: int = config.cell_embedding_size,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        dropout: float = 0.2,
        use_batch_norm: bool = False,
        join_embeddings: Literal['beginning', 'concat', 'sum'] = 'sum',
        train_dataset: PROTAC_Dataset = None,
        val_dataset: PROTAC_Dataset = None,
        test_dataset: PROTAC_Dataset = None,
        disabled_embeddings: List[Literal['smiles', 'poi', 'e3', 'cell']] = [],
        apply_scaling: bool = True,
        extra_optim_params: Optional[dict] = None,
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
            extra_optim_params (dict): Extra parameters for the optimizer
        """
        super().__init__()
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
            use_batch_norm=use_batch_norm,
            disabled_embeddings=[], # NOTE: This is handled in the PROTAC_Dataset classes
        )

        stages = ['train_metrics', 'val_metrics', 'test_metrics']
        self.metrics = nn.ModuleDict({s: MetricCollection({
            'acc': Accuracy(task='binary'),
            'roc_auc': AUROC(task='binary'),
            'precision': Precision(task='binary'),
            'recall': Recall(task='binary'),
            'f1_score': F1Score(task='binary'),
        }, prefix=s.replace('metrics', '')) for s in stages})

        # Misc settings
        self.missing_dataset_error = \
            '''Class variable `{0}` is None. If the model was loaded from a checkpoint, the dataset must be set manually:
            
            model = {1}.load_from_checkpoint('checkpoint.ckpt')
            model.{0} = my_{0}
            '''
        
        # Apply scaling in datasets
        self.scalers = None
        if self.apply_scaling and self.train_dataset is not None:
            self.initialize_scalers()

    def initialize_scalers(self):
        """Initialize or reinitialize scalers based on dataset properties."""
        if self.scalers is None:
            use_single_scaler = self.join_embeddings == 'beginning'
            self.scalers = self.train_dataset.fit_scaling(use_single_scaler)
            self.apply_scalers()

    def apply_scalers(self):
        """Apply scalers to all datasets."""
        use_single_scaler = self.join_embeddings == 'beginning'
        if self.train_dataset:
            self.train_dataset.apply_scaling(self.scalers, use_single_scaler)
        if self.val_dataset:
            self.val_dataset.apply_scaling(self.scalers, use_single_scaler)
        if self.test_dataset:
            self.test_dataset.apply_scaling(self.scalers, use_single_scaler)
    
    def scale_tensor(
            self,
            tensor: torch.Tensor,
            scaler: StandardScaler,
            alpha: float = 1e-10,
    ) -> torch.Tensor:
        """Scale a tensor using a scaler. This is done to avoid using numpy
        arrays (and stay on the same device).
        
        Args:
            tensor (torch.Tensor): The tensor to scale.
            scaler (StandardScaler): The scaler to use.

        Returns:
            torch.Tensor: The scaled tensor.
        """
        tensor = tensor.float()
        if scaler.with_mean:
            tensor -= torch.tensor(scaler.mean_, dtype=tensor.dtype, device=tensor.device)
        if scaler.with_std:
            tensor /= torch.tensor(scaler.scale_, dtype=tensor.dtype, device=tensor.device) + alpha
        return tensor

    def forward(self, poi_emb, e3_emb, cell_emb, smiles_emb, prescaled_embeddings=True, return_embeddings=False):
        if not prescaled_embeddings:
            if self.apply_scaling:
                if self.join_embeddings == 'beginning':
                    embeddings = self.scale_tensor(
                        torch.hstack([smiles_emb, poi_emb, e3_emb, cell_emb]),
                        self.scalers,
                    )
                    smiles_emb = embeddings[:, :self.smiles_emb_dim]
                    poi_emb = embeddings[:, self.smiles_emb_dim:self.smiles_emb_dim+self.poi_emb_dim]
                    e3_emb = embeddings[:, self.smiles_emb_dim+self.poi_emb_dim:self.smiles_emb_dim+2*self.poi_emb_dim]
                    cell_emb = embeddings[:, -self.cell_emb_dim:]
                else:
                    poi_emb = self.scale_tensor(poi_emb, self.scalers['Uniprot'])
                    e3_emb = self.scale_tensor(e3_emb, self.scalers['E3 Ligase Uniprot'])
                    cell_emb = self.scale_tensor(cell_emb, self.scalers['Cell Line Identifier'])
                    smiles_emb = self.scale_tensor(smiles_emb, self.scalers['Smiles'])
        if torch.isnan(poi_emb).any():
            raise ValueError("NaN values found in POI embeddings.")
        if torch.isnan(e3_emb).any():
            raise ValueError("NaN values found in E3 embeddings.")
        if torch.isnan(cell_emb).any():
            raise ValueError("NaN values found in cell embeddings.")
        if torch.isnan(smiles_emb).any():
            raise ValueError("NaN values found in SMILES embeddings.")
        return self.model(poi_emb, e3_emb, cell_emb, smiles_emb, return_embeddings)

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
        # Define optimizer
        if self.extra_optim_params is not None:
            optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, **self.extra_optim_params)
        else:
            optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        # Define LR scheduler
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode='min',
            factor=0.1,
            patience=0,
        )
        # if self.trainer.max_epochs:
        #     total_iters = self.trainer.max_epochs
        # elif self.trainer.max_steps:
        #     total_iters = self.trainer.max_steps
        # else:
        #     total_iters = 20
        # lr_scheduler = optim.lr_scheduler.LinearLR(
        #     optimizer=optimizer,
        #     total_iters=total_iters,
        # )
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
            'interval': 'step',  # or 'epoch'
            'frequency': 1,
            'monitor': 'val_loss',
        }

    def predict_step(self, batch, batch_idx):
        poi_emb = batch['poi_emb']
        e3_emb = batch['e3_emb']
        cell_emb = batch['cell_emb']
        smiles_emb = batch['smiles_emb']
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
    
    def on_save_checkpoint(self, checkpoint):
        """ Serialize the scalers to the checkpoint. """
        checkpoint['scalers'] = pickle.dumps(self.scalers)
    
    def on_load_checkpoint(self, checkpoint):
        """Deserialize the scalers from the checkpoint."""
        if 'scalers' in checkpoint:
            self.scalers = pickle.loads(checkpoint['scalers'])
        else:
            self.scalers = None
        if self.apply_scaling:
            if self.scalers is not None:
                # Re-apply scalers to ensure datasets are scaled
                self.apply_scalers()
            else:
                logging.warning("Scalers not found in checkpoint. Consider re-fitting scalers if necessary.")


def get_confidence_scores(true_ds, y_preds, threshold=0.5):
    # Calculate the likelihood for the false negative: get the mean value of
    # the prediction for the false-positive and false-negatives

    # Convert PyTorch dataset labels to numpy array
    if isinstance(true_ds, PROTAC_Dataset):
        true_vals = np.array([x['active'] for x in true_ds]).flatten()
    elif isinstance(true_ds, torch.Tensor):
        true_vals = true_ds.numpy().flatten()
    elif isinstance(true_ds, np.ndarray):
        true_vals = true_ds.flatten()
    else:
        raise ValueError("Unknown type for true labels.")

    if isinstance(y_preds, torch.Tensor):
        preds = y_preds.numpy().flatten()
    elif isinstance(y_preds, np.ndarray):
        preds = y_preds.flatten()
    else:
        raise ValueError("Unknown type for predictions.")

    logging.info(f"True values: {true_vals}")
    logging.info(f"Predictions: {preds}")

    # Get the indices of the false positives and false negatives
    false_positives = (true_vals == 0) & ((preds > threshold).astype(int) == 1)
    false_negatives = (true_vals == 1) & ((preds > threshold).astype(int) == 0)

    logging.info(f"False positives: {false_positives}")
    logging.info(f"False negatives: {false_negatives}")

    # Get the mean value of the predictions for the false positives and false negatives
    false_positives_mean = preds[false_positives].mean()
    false_negatives_mean = preds[false_negatives].mean()

    return false_positives_mean, false_negatives_mean


# TODO: Use some sort of **kwargs to pass all the parameters to the model...
def train_model(
        protein2embedding: Dict[str, np.ndarray],
        cell2embedding: Dict[str, np.ndarray],
        smiles2fp: Dict[str, np.ndarray],
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: Optional[pd.DataFrame] = None,
        hidden_dim: int = 768,
        batch_size: int = 128,
        learning_rate: float = 2e-5,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        dropout: float = 0.2,
        max_epochs: int = 50,
        use_batch_norm: bool = False,
        join_embeddings: Literal['beginning', 'concat', 'sum'] = 'sum',
        smote_k_neighbors: int = 5,
        apply_scaling: bool = True,
        active_label: str = 'Active',
        fast_dev_run: bool = False,
        use_logger: bool = True,
        logger_save_dir: str = '../logs',
        logger_name: str = 'protac',
        enable_checkpointing: bool = False,
        checkpoint_model_name: str = 'protac',
        disabled_embeddings: List[Literal['smiles', 'poi', 'e3', 'cell']] = [],
        return_predictions: bool = False,
        shuffle_embedding_prob: float = 0.0,
        use_smote: bool = False,
) -> tuple:
    """ Train a PROTAC model using the given datasets and hyperparameters.
    
    Args:
        protein2embedding (dict): A dictionary mapping protein identifiers to embeddings.
        cell2embedding (dict): A dictionary mapping cell line identifiers to embeddings.
        smiles2fp (dict): A dictionary mapping SMILES strings to fingerprints.
        train_df (pd.DataFrame): The training dataframe.
        val_df (pd.DataFrame): The validation dataframe.
        test_df (Optional[pd.DataFrame]): The test dataframe.
        hidden_dim (int): The hidden dimension of the model
        batch_size (int): The batch size
        learning_rate (float): The learning rate
        dropout (float): The dropout rate
        max_epochs (int): The maximum number of epochs
        use_batch_norm (bool): Whether to use batch normalization
        join_embeddings (Literal['beginning', 'concat', 'sum']): How to join the embeddings
        smote_k_neighbors (int): The number of neighbors to use in SMOTE
        use_smote (bool): Whether to use SMOTE
        apply_scaling (bool): Whether to apply scaling to the embeddings
        active_label (str): The name of the active label. Default: 'Active'
        fast_dev_run (bool): Whether to run a fast development run (see PyTorch Lightning documentation)
        use_logger (bool): Whether to use a logger
        logger_save_dir (str): The directory to save the logs
        logger_name (str): The name of the logger
        enable_checkpointing (bool): Whether to enable checkpointing
        checkpoint_model_name (str): The name of the model for checkpointing
        disabled_embeddings (list): List of disabled embeddings. Can be 'poi', 'e3', 'cell', 'smiles'
        return_predictions (bool): Whether to return predictions on the validation and test sets
    
    Returns:
        tuple: The trained model, the trainer, and the metrics over the validation and test sets.
    """
    train_ds, val_ds, test_ds = get_datasets(
        train_df,
        val_df,
        test_df,
        protein2embedding,
        cell2embedding,
        smiles2fp,
        smote_k_neighbors=smote_k_neighbors,
        active_label=active_label,
        disabled_embeddings=disabled_embeddings,
        shuffle_embedding_prob=shuffle_embedding_prob,
    )
    # NOTE: The embeddings dimensions should already match in all sets
    smiles_emb_dim = train_ds.get_smiles_emb_dim()
    poi_emb_dim = train_ds.get_protein_emb_dim()
    e3_emb_dim = train_ds.get_protein_emb_dim()
    cell_emb_dim = train_ds.get_cell_emb_dim()

    loggers = [
        pl.loggers.TensorBoardLogger(
            save_dir=logger_save_dir,
            version=logger_name,
            name=logger_name,
        ),
        pl.loggers.CSVLogger(
            save_dir=logger_save_dir,
            version=logger_name,
            name=logger_name,
        ),
    ]
    callbacks = [
        pl.callbacks.EarlyStopping(
            monitor='train_loss',
            patience=10,
            mode='min',
            verbose=False,
        ),
        pl.callbacks.EarlyStopping(
            monitor='train_acc',
            patience=10,
            mode='max',
            verbose=False,
        ),
        pl.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5, # Original: 5
            mode='min',
            verbose=False,
        ),
        pl.callbacks.EarlyStopping(
            monitor='val_acc',
            patience=10, # Original: 10
            mode='max',
            verbose=False,
        ),
    ]
    if use_logger:
        callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval='step'))
    if enable_checkpointing:
        callbacks.append(pl.callbacks.ModelCheckpoint(
            monitor='val_acc',
            mode='max',
            verbose=False,
            filename=checkpoint_model_name + '-{epoch}-{val_acc:.2f}-{val_roc_auc:.3f}',
        ))
    # Define Trainer
    trainer = pl.Trainer(
        logger=loggers if use_logger else False,
        callbacks=callbacks,
        max_epochs=max_epochs,
        # val_check_interval=0.5,
        fast_dev_run=fast_dev_run,
        enable_model_summary=False,
        enable_checkpointing=enable_checkpointing,
        enable_progress_bar=False,
        devices=1,
        num_nodes=1,
    )
    extra_optim_params = {
        'betas': (beta1, beta2),
        'eps': eps,
    }
    model = PROTAC_Model(
        hidden_dim=hidden_dim,
        smiles_emb_dim=smiles_emb_dim,
        poi_emb_dim=poi_emb_dim,
        e3_emb_dim=e3_emb_dim,
        cell_emb_dim=cell_emb_dim,
        batch_size=batch_size,
        join_embeddings=join_embeddings,
        dropout=dropout,
        use_batch_norm=use_batch_norm,
        learning_rate=learning_rate,
        apply_scaling=apply_scaling,
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds if test_df is not None else None,
        disabled_embeddings=disabled_embeddings,
        extra_optim_params=extra_optim_params,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trainer.fit(model)
    metrics = {}
    # Add train metrics
    train_metrics = {m: v.item() for m, v in trainer.callback_metrics.items() if 'train' in m}
    metrics.update(train_metrics)
    # Add validation metrics
    val_metrics = trainer.validate(model, verbose=False)[0]
    val_metrics = {m: v for m, v in val_metrics.items() if 'val' in m}
    metrics.update(val_metrics)

    # Add test metrics to metrics
    if test_df is not None:
        test_metrics = trainer.test(model, verbose=False)[0]
        test_metrics = {m: v for m, v in test_metrics.items() if 'test' in m}
        metrics.update(test_metrics)
    
    # Return predictions 
    if return_predictions:
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        val_pred = trainer.predict(model, val_dl)
        val_pred = torch.concat(trainer.predict(model, val_dl)).squeeze()

        fp_mean, fn_mean = get_confidence_scores(val_ds, val_pred)
        metrics['val_false_positives_mean'] = fp_mean
        metrics['val_false_negatives_mean'] = fn_mean

        if test_df is not None:
            test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
            test_pred = torch.concat(trainer.predict(model, test_dl)).squeeze()

            fp_mean, fn_mean = get_confidence_scores(test_ds, test_pred)
            metrics['test_false_positives_mean'] = fp_mean
            metrics['test_false_negatives_mean'] = fn_mean

            return model, trainer, metrics, val_pred, test_pred
        return model, trainer, metrics, val_pred
    return model, trainer, metrics


def evaluate_model(
        model: PROTAC_Model,
        trainer: pl.Trainer,
        val_ds: PROTAC_Dataset,
        test_ds: Optional[PROTAC_Dataset] = None,
        batch_size: int = 128,
) -> tuple:
    """ Evaluate a PROTAC model using the given datasets. """
    ret = {}

    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    val_metrics = trainer.validate(model, val_dl, verbose=False)[0]
    val_metrics = {m: v for m, v in val_metrics.items() if 'val' in m}
    # Get predictions on validation set
    val_pred = torch.cat(trainer.predict(model, val_dl)).squeeze()
    ret['val_metrics'] = val_metrics
    ret['val_pred'] = val_pred

    if test_ds is not None:
        test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        test_metrics = trainer.test(model, test_dl, verbose=False)[0]
        test_metrics = {m: v for m, v in test_metrics.items() if 'test' in m}
        # Get predictions on test set
        test_pred = torch.cat(trainer.predict(model, test_dl)).squeeze()
        ret['test_metrics'] = test_metrics
        ret['test_pred'] = test_pred
    
    return ret


def load_model(
        ckpt_path: str,
) -> PROTAC_Model:
    """ Load a PROTAC model from a checkpoint.
    
    Args:
        ckpt_path (str): The path to the checkpoint.
    
    Returns:
        PROTAC_Model: The loaded model.
    """
    # NOTE: The `map_locat` argument is automatically handled in newer versions
    # of PyTorch Lightning, but we keep it here for compatibility with older ones.
    model = PROTAC_Model.load_from_checkpoint(
        ckpt_path,
        map_location=torch.device('cpu') if not torch.cuda.is_available() else None,
    )
    # NOTE: The following is left as example for eventually re-applying scaling
    # with other datasets...
    # if model.apply_scaling:
    #     model.apply_scalers()
    return model.eval()