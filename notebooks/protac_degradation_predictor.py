import os
import pickle
import warnings
import logging
from collections import defaultdict
from typing import Literal, List, Tuple, Optional
import urllib.request

import joblib
import optuna
from optuna.samplers import TPESampler
import h5py
import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from jsonargparse import CLI
from tqdm.auto import tqdm
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedGroupKFold,
)

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
)
from torchmetrics import MetricCollection


# Ignore UserWarning from Matplotlib
warnings.filterwarnings("ignore", ".*FixedLocator*")
# Ignore UserWarning from PyTorch Lightning
warnings.filterwarnings("ignore", ".*does not have many workers.*")

protac_df = pd.read_csv('../data/PROTAC-Degradation-DB.csv')

# Map E3 Ligase Iap to IAP
protac_df['E3 Ligase'] = protac_df['E3 Ligase'].str.replace('Iap', 'IAP')

def is_active(DC50: float, Dmax: float, oring=False, pDC50_threshold=7.0, Dmax_threshold=0.8) -> bool:
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

# ## Load Protein Embeddings

# Protein embeddings downloaded from [Uniprot](https://www.uniprot.org/help/embeddings).
# 
# Please note that running the following cell the first time might take a while.
download_link = "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/embeddings/UP000005640_9606/per-protein.h5"
embeddings_path = "../data/uniprot2embedding.h5"
if not os.path.exists(embeddings_path):
    # Download the file
    print(f'Downloading embeddings from {download_link}')
    urllib.request.urlretrieve(download_link, embeddings_path)

protein_embeddings = {}
with h5py.File("../data/uniprot2embedding.h5", "r") as file:
    uniprots = protac_df['Uniprot'].unique().tolist()
    uniprots += protac_df['E3 Ligase Uniprot'].unique().tolist()
    for i, sequence_id in tqdm(enumerate(uniprots), desc='Loading protein embeddings'):
        try:
            embedding = file[sequence_id][:]
            protein_embeddings[sequence_id] = np.array(embedding)
        except KeyError:
            print(f'KeyError for {sequence_id}')
            protein_embeddings[sequence_id] = np.zeros((1024,))

## Load Cell Embeddings
cell2embedding_filepath = '../data/cell2embedding.pkl'
with open(cell2embedding_filepath, 'rb') as f:
    cell2embedding = pickle.load(f)
print(f'Loaded {len(cell2embedding)} cell lines')

emb_shape = cell2embedding[list(cell2embedding.keys())[0]].shape
# Assign all-zero vectors to cell lines that are not in the embedding file
for cell_line in protac_df['Cell Line Identifier'].unique():
    if cell_line not in cell2embedding:
        cell2embedding[cell_line] = np.zeros(emb_shape)

## Precompute Molecular Fingerprints
fingerprint_size = 224
morgan_fpgen = AllChem.GetMorganGenerator(
    radius=15,
    fpSize=fingerprint_size,
    includeChirality=True,
)

smiles2fp = {}
for smiles in tqdm(protac_df['Smiles'].unique().tolist(), desc='Precomputing fingerprints'):
    # Get the fingerprint as a bit vector
    morgan_fp = morgan_fpgen.GetFingerprint(Chem.MolFromSmiles(smiles))
    smiles2fp[smiles] = morgan_fp

# Count the number of unique SMILES and the number of unique Morgan fingerprints
print(f'Number of unique SMILES: {len(smiles2fp)}')
print(f'Number of unique fingerprints: {len(set([tuple(fp) for fp in smiles2fp.values()]))}')
# Get the list of SMILES with overlapping fingerprints
overlapping_smiles = []
unique_fps = set()
for smiles, fp in smiles2fp.items():
    if tuple(fp) in unique_fps:
        overlapping_smiles.append(smiles)
    else:
        unique_fps.add(tuple(fp))
print(f'Number of SMILES with overlapping fingerprints: {len(overlapping_smiles)}')
print(f'Number of overlapping SMILES in protac_df: {len(protac_df[protac_df["Smiles"].isin(overlapping_smiles)])}')

# Get the pair-wise tanimoto similarity between the PROTAC fingerprints
tanimoto_matrix = defaultdict(list)
for i, smiles1 in enumerate(tqdm(protac_df['Smiles'].unique(), desc='Computing Tanimoto similarity')):
    fp1 = smiles2fp[smiles1]
    # TODO: Use BulkTanimotoSimilarity for better performance
    for j, smiles2 in enumerate(protac_df['Smiles'].unique()):
        if j < i:
            continue
        fp2 = smiles2fp[smiles2]
        tanimoto_dist = DataStructs.TanimotoSimilarity(fp1, fp2)
        tanimoto_matrix[smiles1].append(tanimoto_dist)
avg_tanimoto = {k: np.mean(v) for k, v in tanimoto_matrix.items()}
protac_df['Avg Tanimoto'] = protac_df['Smiles'].map(avg_tanimoto)

smiles2fp = {s: np.array(fp) for s, fp in smiles2fp.items()}


class PROTAC_Dataset(Dataset):
    def __init__(
        self,
        protac_df,
        protein_embeddings=protein_embeddings,
        cell2embedding=cell2embedding,
        smiles2fp=smiles2fp,
        use_smote=False,
        oversampler=None,
        active_label='Active',
        include_mol_graphs=False,
    ):
        """ Initialize the PROTAC dataset

        Args:
            protac_df (pd.DataFrame): The PROTAC dataframe
            protein_embeddings (dict): Dictionary of protein embeddings
            cell2embedding (dict): Dictionary of cell line embeddings
            smiles2fp (dict): Dictionary of SMILES to fingerprint
            use_smote (bool): Whether to use SMOTE for oversampling
            use_ored_activity (bool): Whether to use the 'Active - OR' column
        """
        # Filter out examples with NaN in active_col column
        self.data = protac_df  # [~protac_df[active_col].isna()]
        self.protein_embeddings = protein_embeddings
        self.cell2embedding = cell2embedding
        self.smiles2fp = smiles2fp
        self.active_label = active_label
        self.include_mol_graphs = include_mol_graphs

        self.smiles_emb_dim = smiles2fp[list(smiles2fp.keys())[0]].shape[0]
        self.protein_emb_dim = protein_embeddings[list(
            protein_embeddings.keys())[0]].shape[0]
        self.cell_emb_dim = cell2embedding[list(
            cell2embedding.keys())[0]].shape[0]

        # Look up the embeddings
        self.data = pd.DataFrame({
            'Smiles': self.data['Smiles'].apply(lambda x: smiles2fp[x].astype(np.float32)).tolist(),
            'Uniprot': self.data['Uniprot'].apply(lambda x: protein_embeddings[x].astype(np.float32)).tolist(),
            'E3 Ligase Uniprot': self.data['E3 Ligase Uniprot'].apply(lambda x: protein_embeddings[x].astype(np.float32)).tolist(),
            'Cell Line Identifier': self.data['Cell Line Identifier'].apply(lambda x: cell2embedding[x].astype(np.float32)).tolist(),
            self.active_label: self.data[self.active_label].astype(np.float32).tolist(),
        })

        # Apply SMOTE
        self.use_smote = use_smote
        self.oversampler = oversampler
        if self.use_smote:
            self.apply_smote()

    def apply_smote(self):
        # Prepare the dataset for SMOTE
        features = []
        labels = []
        for _, row in self.data.iterrows():
            features.append(np.hstack([
                row['Smiles'],
                row['Uniprot'],
                row['E3 Ligase Uniprot'],
                row['Cell Line Identifier'],
            ]))
            labels.append(row[self.active_label])

        # Convert to numpy array
        features = np.array(features).astype(np.float32)
        labels = np.array(labels).astype(np.float32)

        # Initialize SMOTE and fit
        if self.oversampler is None:
            oversampler = SMOTE(random_state=42)
        else:
            oversampler = self.oversampler
        features_smote, labels_smote = oversampler.fit_resample(features, labels)

        # Separate the features back into their respective embeddings
        smiles_embs = features_smote[:, :self.smiles_emb_dim]
        poi_embs = features_smote[:,
                                  self.smiles_emb_dim:self.smiles_emb_dim+self.protein_emb_dim]
        e3_embs = features_smote[:, self.smiles_emb_dim +
                                 self.protein_emb_dim:self.smiles_emb_dim+2*self.protein_emb_dim]
        cell_embs = features_smote[:, -self.cell_emb_dim:]

        # Reconstruct the dataframe with oversampled data
        df_smote = pd.DataFrame({
            'Smiles': list(smiles_embs),
            'Uniprot': list(poi_embs),
            'E3 Ligase Uniprot': list(e3_embs),
            'Cell Line Identifier': list(cell_embs),
            self.active_label: labels_smote
        })
        self.data = df_smote

    def fit_scaling(self, use_single_scaler=False, **scaler_kwargs) -> dict:
        """ Fit the scalers for the data.

        Returns:
            dict: The fitted scalers.
        """
        if use_single_scaler:
            scaler = StandardScaler(**scaler_kwargs)
            embeddings = np.hstack([
                np.array(self.data['Smiles'].tolist()),
                np.array(self.data['Uniprot'].tolist()),
                np.array(self.data['E3 Ligase Uniprot'].tolist()),
                np.array(self.data['Cell Line Identifier'].tolist()),
            ])
            scaler.fit(embeddings)
            return scaler
        else:
            scalers = {}
            scalers['Smiles'] = StandardScaler(**scaler_kwargs)
            scalers['Uniprot'] = StandardScaler(**scaler_kwargs)
            scalers['E3 Ligase Uniprot'] = StandardScaler(**scaler_kwargs)
            scalers['Cell Line Identifier'] = StandardScaler(**scaler_kwargs)

            scalers['Smiles'].fit(np.stack(self.data['Smiles'].to_numpy()))
            scalers['Uniprot'].fit(np.stack(self.data['Uniprot'].to_numpy()))
            scalers['E3 Ligase Uniprot'].fit(np.stack(self.data['E3 Ligase Uniprot'].to_numpy()))
            scalers['Cell Line Identifier'].fit(np.stack(self.data['Cell Line Identifier'].to_numpy()))

            return scalers

    def apply_scaling(self, scalers: dict, use_single_scaler=False):
        """ Apply scaling to the data.

        Args:
            scalers (dict): The scalers for each feature.
        """
        if use_single_scaler:
            embeddings = np.hstack([
                np.array(self.data['Smiles'].tolist()),
                np.array(self.data['Uniprot'].tolist()),
                np.array(self.data['E3 Ligase Uniprot'].tolist()),
                np.array(self.data['Cell Line Identifier'].tolist()),
            ])
            scaled_embeddings = scalers.transform(embeddings)
            self.data = pd.DataFrame({
                'Smiles': list(scaled_embeddings[:, :self.smiles_emb_dim]),
                'Uniprot': list(scaled_embeddings[:, self.smiles_emb_dim:self.smiles_emb_dim+self.protein_emb_dim]),
                'E3 Ligase Uniprot': list(scaled_embeddings[:, self.smiles_emb_dim+self.protein_emb_dim:self.smiles_emb_dim+2*self.protein_emb_dim]),
                'Cell Line Identifier': list(scaled_embeddings[:, -self.cell_emb_dim:]),
                self.active_label: self.data[self.active_label]
            })
        else:
            self.data['Smiles'] = self.data['Smiles'].apply(lambda x: scalers['Smiles'].transform(x[np.newaxis, :])[0])
            self.data['Uniprot'] = self.data['Uniprot'].apply(lambda x: scalers['Uniprot'].transform(x[np.newaxis, :])[0])
            self.data['E3 Ligase Uniprot'] = self.data['E3 Ligase Uniprot'].apply(lambda x: scalers['E3 Ligase Uniprot'].transform(x[np.newaxis, :])[0])
            self.data['Cell Line Identifier'] = self.data['Cell Line Identifier'].apply(lambda x: scalers['Cell Line Identifier'].transform(x[np.newaxis, :])[0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        elem = {
            'smiles_emb': self.data['Smiles'].iloc[idx],
            'poi_emb': self.data['Uniprot'].iloc[idx],
            'e3_emb': self.data['E3 Ligase Uniprot'].iloc[idx],
            'cell_emb': self.data['Cell Line Identifier'].iloc[idx],
            'active': self.data[self.active_label].iloc[idx],
        }
        return elem


class PROTAC_Model(pl.LightningModule):

    def __init__(
        self,
        hidden_dim: int,
        smiles_emb_dim: int = fingerprint_size,
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
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: Optional[pd.DataFrame] = None,
        hidden_dim: int = 768,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        dropout: float = 0.2,
        max_epochs: int = 50,
        smiles_emb_dim: int = fingerprint_size,
        join_embeddings: Literal['beginning', 'concat', 'sum'] = 'concat',
        smote_k_neighbors:int = 5,
        use_smote: bool = True,
        apply_scaling: bool = False,
        active_label:str = 'Active',
        fast_dev_run: bool = False,
        use_logger: bool = True,
        logger_name: str = 'protac',
        disabled_embeddings: List[str] = [],
) -> tuple:
    """ Train a PROTAC model using the given datasets and hyperparameters.
    
    Args:
        train_df (pd.DataFrame): The training set.
        val_df (pd.DataFrame): The validation set.
        test_df (pd.DataFrame): The test set.
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
        protein_embeddings,
        cell2embedding,
        smiles2fp,
        use_smote=use_smote,
        oversampler=oversampler if use_smote else None,
        active_label=active_label,
    )
    val_ds = PROTAC_Dataset(
        val_df,
        protein_embeddings,
        cell2embedding,
        smiles2fp,
        active_label=active_label,
    )
    if test_df is not None:
        test_ds = PROTAC_Dataset(
            test_df,
            protein_embeddings,
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

# Setup hyperparameter optimization:

def objective(
        trial: optuna.Trial,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        hidden_dim_options: List[int] = [256, 512, 768],
        batch_size_options: List[int] = [8, 16, 32],
        learning_rate_options: Tuple[float, float] = (1e-5, 1e-3),
        smote_k_neighbors_options: List[int] = list(range(3, 16)),
        dropout_options: Tuple[float, float] = (0.1, 0.5),
        fast_dev_run: bool = False,
        active_label: str = 'Active',
        disabled_embeddings: List[str] = [],
) -> float:
    """ Objective function for hyperparameter optimization.
    
    Args:
        trial (optuna.Trial): The Optuna trial object.
        train_df (pd.DataFrame): The training set.
        val_df (pd.DataFrame): The validation set.
        hidden_dim_options (List[int]): The hidden dimension options.
        batch_size_options (List[int]): The batch size options.
        learning_rate_options (Tuple[float, float]): The learning rate options.
        smote_k_neighbors_options (List[int]): The SMOTE k neighbors options.
        dropout_options (Tuple[float, float]): The dropout options.
        fast_dev_run (bool): Whether to run a fast development run.
        active_label (str): The active label column.
        disabled_embeddings (List[str]): The list of disabled embeddings.
    """
    # Generate the hyperparameters
    hidden_dim = trial.suggest_categorical('hidden_dim', hidden_dim_options)
    batch_size = trial.suggest_categorical('batch_size', batch_size_options)
    learning_rate = trial.suggest_float('learning_rate', *learning_rate_options, log=True)
    join_embeddings = trial.suggest_categorical('join_embeddings', ['beginning', 'concat', 'sum'])
    smote_k_neighbors = trial.suggest_categorical('smote_k_neighbors', smote_k_neighbors_options)
    use_smote = trial.suggest_categorical('use_smote', [True, False])
    apply_scaling = trial.suggest_categorical('apply_scaling', [True, False])
    dropout = trial.suggest_float('dropout', *dropout_options)

    # Train the model with the current set of hyperparameters
    _, _, metrics = train_model(
        train_df,
        val_df,
        hidden_dim=hidden_dim,
        batch_size=batch_size,
        join_embeddings=join_embeddings,
        learning_rate=learning_rate,
        dropout=dropout,
        max_epochs=100,
        smote_k_neighbors=smote_k_neighbors,
        apply_scaling=apply_scaling,
        use_smote=use_smote,
        use_logger=False,
        fast_dev_run=fast_dev_run,
        active_label=active_label,
        disabled_embeddings=disabled_embeddings,
    )

    # Metrics is a dictionary containing at least the validation loss
    val_loss = metrics['val_loss']
    val_acc = metrics['val_acc']
    val_roc_auc = metrics['val_roc_auc']
    
    # Optuna aims to minimize the objective
    return val_loss - val_acc - val_roc_auc


def hyperparameter_tuning_and_training(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        fast_dev_run: bool = False,
        n_trials: int = 50,
        logger_name: str = 'protac_hparam_search',
        active_label: str = 'Active',
        disabled_embeddings: List[str] = [],
        study_filename: Optional[str] = None,
) -> tuple:
    """ Hyperparameter tuning and training of a PROTAC model.
    
    Args:
        train_df (pd.DataFrame): The training set.
        val_df (pd.DataFrame): The validation set.
        test_df (pd.DataFrame): The test set.
        fast_dev_run (bool): Whether to run a fast development run.
        n_trials (int): The number of hyperparameter optimization trials.
        logger_name (str): The name of the logger.
        active_label (str): The active label column.
        disabled_embeddings (List[str]): The list of disabled embeddings.

    Returns:
        tuple: The trained model, the trainer, and the best metrics.
    """
    # Define the search space
    hidden_dim_options = [256, 512, 768]
    batch_size_options = [8, 16, 32]
    learning_rate_options = (1e-5, 1e-3) # min and max values for loguniform distribution
    smote_k_neighbors_options = list(range(3, 16))

    # Set the verbosity of Optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # Create an Optuna study object
    sampler = TPESampler(seed=42, multivariate=True)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(
        lambda trial: objective(
            trial,
            train_df,
            val_df,
            hidden_dim_options=hidden_dim_options,
            batch_size_options=batch_size_options,
            learning_rate_options=learning_rate_options,
            smote_k_neighbors_options=smote_k_neighbors_options,
            fast_dev_run=fast_dev_run,
            active_label=active_label,
            disabled_embeddings=disabled_embeddings,
        ),
        n_trials=n_trials,
    )

    if study_filename:
        joblib.dump(study, study_filename)

    # Retrain the model with the best hyperparameters
    model, trainer, metrics = train_model(
        train_df,
        val_df,
        test_df,
        use_logger=True,
        logger_name=logger_name,
        fast_dev_run=fast_dev_run,
        active_label=active_label,
        disabled_embeddings=disabled_embeddings,
        **study.best_params,
    )

    # Report the best hyperparameters found
    metrics.update({f'hparam_{k}': v for k, v in study.best_params.items()})

    # Return the best metrics
    return model, trainer, metrics


def main(
    active_col: str = 'Active (Dmax 0.6, pDC50 6.0)',
    n_trials: int = 50,
    fast_dev_run: bool = False,
    test_split: float = 0.2,
    cv_n_splits: int = 5,
):
    """ Train a PROTAC model using the given datasets and hyperparameters.
    
    Args:
        use_ored_activity (bool): Whether to use the 'Active - OR' column.
        n_trials (int): The number of hyperparameter optimization trials.
        n_splits (int): The number of cross-validation splits.
        fast_dev_run (bool): Whether to run a fast development run.
    """
    ## Set the Column to Predict
    active_name = active_col.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')

    # Get Dmax_threshold from the active_col
    Dmax_threshold = float(active_col.split('Dmax')[1].split(',')[0].strip('(').strip(')').strip())
    pDC50_threshold = float(active_col.split('pDC50')[1].strip('(').strip(')').strip())

    protac_df[active_col] = protac_df.apply(
        lambda x: is_active(x['DC50 (nM)'], x['Dmax (%)'], pDC50_threshold=pDC50_threshold, Dmax_threshold=Dmax_threshold), axis=1
    )

    ## Test Sets

    test_indeces = {}

    ### Random Split

    # Randomly select 20% of the active PROTACs as the test set
    active_df = protac_df[protac_df[active_col].notna()].copy()
    test_df = active_df.sample(frac=test_split, random_state=42)
    test_indeces['random'] = test_df.index

    ### E3-based Split

    encoder = OrdinalEncoder()
    protac_df['E3 Group'] = encoder.fit_transform(protac_df[['E3 Ligase']]).astype(int)
    active_df = protac_df[protac_df[active_col].notna()].copy()
    test_df = active_df[(active_df['E3 Ligase'] != 'VHL') & (active_df['E3 Ligase'] != 'CRBN')]
    test_indeces['e3_ligase'] = test_df.index

    ### Tanimoto-based Split

    n_bins_tanimoto = 200
    tanimoto_groups = pd.cut(protac_df['Avg Tanimoto'], bins=n_bins_tanimoto).copy()
    encoder = OrdinalEncoder()
    protac_df['Tanimoto Group'] = encoder.fit_transform(tanimoto_groups.values.reshape(-1, 1)).astype(int)
    active_df = protac_df[protac_df[active_col].notna()].copy()

    test_df = []
    # For each group, get the number of active and inactive entries. Then, add those
    # entries to the test_df if: 1) the test_df lenght + the group entries is less
    # 20% of the active_df lenght, and 2) the percentage of True and False entries
    # in the active_col in test_df is roughly 50%.
    # Start the loop from the groups containing the smallest number of entries.
    for group in reversed(active_df['Tanimoto Group'].value_counts().index):
        group_df = active_df[active_df['Tanimoto Group'] == group]
        if test_df == []:
            test_df.append(group_df)
            continue
        
        num_entries = len(group_df)
        num_active_group = group_df[active_col].sum()
        num_inactive_group = num_entries - num_active_group

        tmp_test_df = pd.concat(test_df)
        num_entries_test = len(tmp_test_df)
        num_active_test = tmp_test_df[active_col].sum()
        num_inactive_test = num_entries_test - num_active_test
        
        # Check if the group entries can be added to the test_df
        if num_entries_test + num_entries < test_split * len(active_df):
            # Add anything at the beggining
            if num_entries_test + num_entries < test_split / 2 * len(active_df):
                test_df.append(group_df)
                continue
            # Be more selective and make sure that the percentage of active and
            # inactive is balanced
            if (num_active_group + num_active_test) / (num_entries_test + num_entries) < 0.6:
                if (num_inactive_group + num_inactive_test) / (num_entries_test + num_entries) < 0.6:
                    test_df.append(group_df)
    test_df = pd.concat(test_df)
    # Save to global dictionary of test indeces
    test_indeces['tanimoto'] = test_df.index

    ### Target-based Split

    encoder = OrdinalEncoder()
    protac_df['Uniprot Group'] = encoder.fit_transform(protac_df[['Uniprot']]).astype(int)
    active_df = protac_df[protac_df[active_col].notna()].copy()

    test_df = []
    # For each group, get the number of active and inactive entries. Then, add those
    # entries to the test_df if: 1) the test_df lenght + the group entries is less
    # 20% of the active_df lenght, and 2) the percentage of True and False entries
    # in the active_col in test_df is roughly 50%.
    # Start the loop from the groups containing the smallest number of entries.
    for group in reversed(active_df['Uniprot'].value_counts().index):
        group_df = active_df[active_df['Uniprot'] == group]
        if test_df == []:
            test_df.append(group_df)
            continue
        
        num_entries = len(group_df)
        num_active_group = group_df[active_col].sum()
        num_inactive_group = num_entries - num_active_group

        tmp_test_df = pd.concat(test_df)
        num_entries_test = len(tmp_test_df)
        num_active_test = tmp_test_df[active_col].sum()
        num_inactive_test = num_entries_test - num_active_test
        
        # Check if the group entries can be added to the test_df
        if num_entries_test + num_entries < test_split * len(active_df):
            # Add anything at the beggining
            if num_entries_test + num_entries < test_split / 2 * len(active_df):
                test_df.append(group_df)
                continue
            # Be more selective and make sure that the percentage of active and
            # inactive is balanced
            if (num_active_group + num_active_test) / (num_entries_test + num_entries) < 0.6:
                if (num_inactive_group + num_inactive_test) / (num_entries_test + num_entries) < 0.6:
                    test_df.append(group_df)
    test_df = pd.concat(test_df)
    # Save to global dictionary of test indeces
    test_indeces['uniprot'] = test_df.index

    ## Cross-Validation Training
    
    # Make directory ../reports if it does not exist
    if not os.path.exists('../reports'):
        os.makedirs('../reports')

    report = []
    for split_type, indeces in test_indeces.items():
        active_df = protac_df[protac_df[active_col].notna()].copy()
        test_df = active_df.loc[indeces]
        train_val_df = active_df[~active_df.index.isin(test_df.index)]
        
        if split_type == 'random':
            kf = StratifiedKFold(n_splits=cv_n_splits, shuffle=True, random_state=42)
            group = None
        elif split_type == 'e3_ligase':
            kf = StratifiedKFold(n_splits=cv_n_splits, shuffle=True, random_state=42)
            group = train_val_df['E3 Group'].to_numpy()
        elif split_type == 'tanimoto':
            kf = StratifiedGroupKFold(n_splits=cv_n_splits, shuffle=True, random_state=42)
            group = train_val_df['Tanimoto Group'].to_numpy()
        elif split_type == 'uniprot':
            kf = StratifiedGroupKFold(n_splits=cv_n_splits, shuffle=True, random_state=42)
            group = train_val_df['Uniprot Group'].to_numpy()
        # Start the CV over the folds
        X = train_val_df.drop(columns=active_col)
        y = train_val_df[active_col].tolist()
        for k, (train_index, val_index) in enumerate(kf.split(X, y, group)):
            print('-' * 100)
            print(f'Starting CV for group type: {split_type}, fold: {k}')
            print('-' * 100)
            train_df = train_val_df.iloc[train_index]
            val_df = train_val_df.iloc[val_index]

            leaking_uniprot = list(set(train_df['Uniprot']).intersection(set(val_df['Uniprot'])))
            leaking_smiles = list(set(train_df['Smiles']).intersection(set(val_df['Smiles'])))

            stats = {
                'fold': k,
                'split_type': split_type,
                'train_len': len(train_df),
                'val_len': len(val_df),
                'train_perc': len(train_df) / len(train_val_df),
                'val_perc': len(val_df) / len(train_val_df),
                'train_active_perc': train_df[active_col].sum() / len(train_df),
                'train_inactive_perc': (len(train_df) - train_df[active_col].sum()) / len(train_df),
                'val_active_perc': val_df[active_col].sum() / len(val_df),
                'val_inactive_perc': (len(val_df) - val_df[active_col].sum()) / len(val_df),
                'test_active_perc': test_df[active_col].sum() / len(test_df),
                'test_inactive_perc': (len(test_df) - test_df[active_col].sum()) / len(test_df),
                'num_leaking_uniprot': len(leaking_uniprot),
                'num_leaking_smiles': len(leaking_smiles),
                'train_leaking_uniprot_perc': len(train_df[train_df['Uniprot'].isin(leaking_uniprot)]) / len(train_df),
                'train_leaking_smiles_perc': len(train_df[train_df['Smiles'].isin(leaking_smiles)]) / len(train_df),
            }
            if split_type != 'random':
                stats['train_unique_groups'] = len(np.unique(group[train_index]))
                stats['val_unique_groups'] = len(np.unique(group[val_index]))
            
            # Train and evaluate the model
            model, trainer, metrics = hyperparameter_tuning_and_training(
                train_df,
                val_df,
                test_df,
                fast_dev_run=fast_dev_run,
                n_trials=n_trials,
                logger_name=f'protac_{active_name}_{split_type}_fold_{k}_test_split_{test_split}',
                active_label=active_col,
                study_filename=f'../reports/study_{active_name}_{split_type}_fold_{k}_test_split_{test_split}.pkl',
            )
            hparams = {p.strip('hparam_'): v for p, v in stats.items() if p.startswith('hparam_')}
            stats.update(metrics)
            report.append(stats.copy())
            del model
            del trainer

            # Ablation study: disable embeddings at a time
            for disabled_embeddings in [['poi'], ['cell'], ['smiles'], ['e3', 'cell'], ['poi', 'e3', 'cell']]:
                print('-' * 100)
                print(f'Ablation study with disabled embeddings: {disabled_embeddings}')
                print('-' * 100)
                stats['disabled_embeddings'] = 'disabled ' + ' '.join(disabled_embeddings)
                model, trainer, metrics = train_model(
                    train_df,
                    val_df,
                    test_df,
                    fast_dev_run=fast_dev_run,
                    logger_name=f'protac_{active_name}_{split_type}_fold_{k}_disabled-{"-".join(disabled_embeddings)}',
                    active_label=active_col,
                    disabled_embeddings=disabled_embeddings,
                    **hparams,
                )
                stats.update(metrics)
                report.append(stats.copy())
                del model
                del trainer

        report_df = pd.DataFrame(report)
        report_df.to_csv(
            f'../reports/cv_report_hparam_search_{cv_n_splits}-splits_{active_name}_test_split_{test_split}.csv',
            index=False,
        )


if __name__ == '__main__':
    cli = CLI(main)