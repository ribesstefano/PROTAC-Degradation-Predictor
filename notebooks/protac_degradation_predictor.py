import optuna
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from collections import defaultdict

import h5py
import numpy as np
from tqdm.auto import tqdm

import os
import urllib.request

from sklearn.preprocessing import StandardScaler

# ## Define Torch Dataset

from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics import (
    Accuracy,
    AUROC,
    Precision,
    Recall,
    F1Score,
)
from torchmetrics import MetricCollection

import pickle

from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedGroupKFold,
)
from sklearn.preprocessing import OrdinalEncoder


protac_df = pd.read_csv('../data/PROTAC-Degradation-DB.csv')
protac_df.head()

# Get the unique Article IDs of the entries with NaN values in the Active column
nan_active = protac_df[protac_df['Active'].isna()]['Article DOI'].unique()
nan_active

# Map E3 Ligase Iap to IAP
protac_df['E3 Ligase'] = protac_df['E3 Ligase'].str.replace('Iap', 'IAP')

cells = sorted(protac_df['Cell Type'].dropna().unique().tolist())
print(f'Number of non-cleaned cell lines: {len(cells)}')

cells = sorted(protac_df['Cell Line Identifier'].dropna().unique().tolist())
print(f'Number of cleaned cell lines: {len(cells)}')

unlabeled_df = protac_df[protac_df['Active'].isna()]
print(f'Number of compounds in test set: {len(unlabeled_df)}')

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
    print(f"number of entries: {len(file.items()):,}")
    uniprots = protac_df['Uniprot'].unique().tolist()
    uniprots += protac_df['E3 Ligase Uniprot'].unique().tolist()
    for i, sequence_id in tqdm(enumerate(uniprots), desc='Loading protein embeddings'):
        try:
            embedding = file[sequence_id][:]
            protein_embeddings[sequence_id] = np.array(embedding)
            if i < 10:
                print(
                    f"\tid: {sequence_id}, "
                    f"\tembeddings shape: {embedding.shape}, "
                    f"\tembeddings mean: {np.array(embedding).mean()}"
                )
        except KeyError:
            print(f'KeyError for {sequence_id}')
            protein_embeddings[sequence_id] = np.zeros((1024,))

# ## Load Cell Embeddings


cell2embedding_filepath = '../data/cell2embedding.pkl'
with open(cell2embedding_filepath, 'rb') as f:
    cell2embedding = pickle.load(f)
print(f'Loaded {len(cell2embedding)} cell lines')

emb_shape = cell2embedding[list(cell2embedding.keys())[0]].shape
# Assign all-zero vectors to cell lines that are not in the embedding file
for cell_line in protac_df['Cell Line Identifier'].unique():
    if cell_line not in cell2embedding:
        cell2embedding[cell_line] = np.zeros(emb_shape)

# ## Precompute Molecular Fingerprints
        
morgan_fpgen = AllChem.GetMorganGenerator(
    radius=15,
    fpSize=1024,
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
    # TODO: Use BulkTanimotoSimilarity
    for j, smiles2 in enumerate(protac_df['Smiles'].unique()):
        if j < i:
            continue
        fp2 = smiles2fp[smiles2]
        tanimoto_dist = DataStructs.TanimotoSimilarity(fp1, fp2)
        tanimoto_matrix[smiles1].append(tanimoto_dist)
avg_tanimoto = {k: np.mean(v) for k, v in tanimoto_matrix.items()}
protac_df['Avg Tanimoto'] = protac_df['Smiles'].map(avg_tanimoto)

smiles2fp = {s: np.array(fp) for s, fp in smiles2fp.items()}

# ## Set the Column to Predict

active_col = 'Active'
# active_col = 'Active - OR'


class PROTAC_Dataset(Dataset):
    def __init__(
        self,
        protac_df,
        protein_embeddings=protein_embeddings,
        cell2embedding=cell2embedding,
        smiles2fp=smiles2fp,
        use_smote=False,
        oversampler=None,
        use_ored_activity=False,
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
        # Filter out examples with NaN in 'Active' column
        self.data = protac_df  # [~protac_df['Active'].isna()]
        self.protein_embeddings = protein_embeddings
        self.cell2embedding = cell2embedding
        self.smiles2fp = smiles2fp

        self.smiles_emb_dim = smiles2fp[list(smiles2fp.keys())[0]].shape[0]
        self.protein_emb_dim = protein_embeddings[list(
            protein_embeddings.keys())[0]].shape[0]
        self.cell_emb_dim = cell2embedding[list(
            cell2embedding.keys())[0]].shape[0]

        self.active_label = 'Active - OR' if use_ored_activity else 'Active'

        self.use_smote = use_smote
        self.oversampler = oversampler
        # Apply SMOTE
        if self.use_smote:
            self.apply_smote()

    def apply_smote(self):
        # Prepare the dataset for SMOTE
        features = []
        labels = []
        for _, row in self.data.iterrows():
            smiles_emb = smiles2fp[row['Smiles']]
            poi_emb = protein_embeddings[row['Uniprot']]
            e3_emb = protein_embeddings[row['E3 Ligase Uniprot']]
            cell_emb = cell2embedding[row['Cell Line Identifier']]
            features.append(np.hstack([
                smiles_emb.astype(np.float32),
                poi_emb.astype(np.float32),
                e3_emb.astype(np.float32),
                cell_emb.astype(np.float32),
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.use_smote:
            # NOTE: We do not need to look up the embeddings anymore
            elem = {
                'smiles_emb': self.data['Smiles'].iloc[idx],
                'poi_emb': self.data['Uniprot'].iloc[idx],
                'e3_emb': self.data['E3 Ligase Uniprot'].iloc[idx],
                'cell_emb': self.data['Cell Line Identifier'].iloc[idx],
                'active': self.data[self.active_label].iloc[idx],
            }
        else:
            elem = {
                'smiles_emb': self.smiles2fp[self.data['Smiles'].iloc[idx]].astype(np.float32),
                'poi_emb': self.protein_embeddings[self.data['Uniprot'].iloc[idx]].astype(np.float32),
                'e3_emb': self.protein_embeddings[self.data['E3 Ligase Uniprot'].iloc[idx]].astype(np.float32),
                'cell_emb': self.cell2embedding[self.data['Cell Line Identifier'].iloc[idx]].astype(np.float32),
                'active': 1. if self.data[self.active_label].iloc[idx] else 0.,
            }
        return elem

# Ignore UserWarning from PyTorch Lightning
warnings.filterwarnings("ignore", ".*does not have many workers.*")

class PROTAC_Model(pl.LightningModule):

    def __init__(
        self,
        hidden_dim,
        smiles_emb_dim=1024,
        poi_emb_dim=1024,
        e3_emb_dim=1024,
        cell_emb_dim=768,
        batch_size=32,
        learning_rate=1e-3,
        dropout=0.2,
        train_dataset=None,
        val_dataset=None,
        test_dataset=None,
        disabled_embeddings=[],
    ):
        super().__init__()
        self.poi_emb_dim = poi_emb_dim
        self.e3_emb_dim = e3_emb_dim
        self.cell_emb_dim = cell_emb_dim
        self.smiles_emb_dim = smiles_emb_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.disabled_embeddings = disabled_embeddings
        # Set our init args as class attributes
        self.__dict__.update(locals())  # Add arguments as attributes
        # Save the arguments passed to init
        ignore_args_as_hyperparams = [
            'train_dataset',
            'test_dataset',
            'val_dataset',
        ]
        self.save_hyperparameters(ignore=ignore_args_as_hyperparams)

        if 'poi' not in self.disabled_embeddings:
            self.poi_emb = nn.Linear(poi_emb_dim, hidden_dim)
            # # Set the POI surrogate model as a Sequential model
            # self.poi_emb = nn.Sequential(
            #     nn.Linear(poi_emb_dim, hidden_dim),
            #     nn.GELU(),
            #     nn.Dropout(p=dropout),
            #     nn.Linear(hidden_dim, hidden_dim),
            #     # nn.ReLU(),
            #     # nn.Dropout(p=dropout),
            # )
        if 'e3' not in self.disabled_embeddings:
            self.e3_emb = nn.Linear(e3_emb_dim, hidden_dim)
            # self.e3_emb = nn.Sequential(
            #     nn.Linear(e3_emb_dim, hidden_dim),
            #     # nn.ReLU(),
            #     nn.Dropout(p=dropout),
            #     # nn.Linear(hidden_dim, hidden_dim),
            #     # nn.ReLU(),
            #     # nn.Dropout(p=dropout),
            # )
        if 'cell' not in self.disabled_embeddings:
            self.cell_emb = nn.Linear(cell_emb_dim, hidden_dim)
            # self.cell_emb = nn.Sequential(
            #     nn.Linear(cell_emb_dim, hidden_dim),
            #     # nn.ReLU(),
            #     nn.Dropout(p=dropout),
            #     # nn.Linear(hidden_dim, hidden_dim),
            #     # nn.ReLU(),
            #     # nn.Dropout(p=dropout),
            # )
        if 'smiles' not in self.disabled_embeddings:
            self.smiles_emb = nn.Linear(smiles_emb_dim, hidden_dim)
            # self.smiles_emb = nn.Sequential(
            #     nn.Linear(smiles_emb_dim, hidden_dim),
            #     # nn.ReLU(),
            #     nn.Dropout(p=dropout),
            #     # nn.Linear(hidden_dim, hidden_dim),
            #     # nn.ReLU(),
            #     # nn.Dropout(p=dropout),
            # )

        self.fc1 = nn.Linear(
            hidden_dim * (4 - len(self.disabled_embeddings)), hidden_dim)
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

    def forward(self, poi_emb, e3_emb, cell_emb, smiles_emb):
        embeddings = []
        if 'poi' not in self.disabled_embeddings:
            embeddings.append(self.poi_emb(poi_emb))
        if 'e3' not in self.disabled_embeddings:
            embeddings.append(self.e3_emb(e3_emb))
        if 'cell' not in self.disabled_embeddings:
            embeddings.append(self.cell_emb(cell_emb))
        if 'smiles' not in self.disabled_embeddings:
            embeddings.append(self.smiles_emb(smiles_emb))
        x = torch.cat(embeddings, dim=1)
        x = self.dropout(F.gelu(self.fc1(x)))
        x = self.dropout(F.gelu(self.fc2(x)))
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

# ## Test Sets

# We want a different test set per Cross-Validation (CV) experiment (see further down). We are interested in three scenarios:
# * Randomly splitting the data into training and test sets. Hence, the test st shall contain unique SMILES and Uniprots
# * Splitting the data according to their Uniprot. Hence, the test set shall contain unique Uniprots
# * Splitting the data according to their SMILES, _i.e._, the test set shall contain unique SMILES

test_indeces = {}

# Isolating the unique SMILES and Uniprots:

active_df = protac_df[protac_df[active_col].notna()].copy()

# Get the unique SMILES and Uniprot
unique_smiles = active_df['Smiles'].value_counts() == 1
unique_uniprot = active_df['Uniprot'].value_counts() == 1
print(f'Number of unique SMILES: {unique_smiles.sum()}')
print(f'Number of unique Uniprot: {unique_uniprot.sum()}')
# Sample 1% of the len(active_df) from unique SMILES and Uniprot and get the
# indices for a test set
n = int(0.05 * len(active_df)) // 2
unique_smiles = unique_smiles[unique_smiles].sample(n=n, random_state=42)
# unique_uniprot = unique_uniprot[unique_uniprot].sample(n=, random_state=42)
unique_indices = active_df[
    active_df['Smiles'].isin(unique_smiles.index) &
    active_df['Uniprot'].isin(unique_uniprot.index)
].index
print(f'Number of unique indices: {len(unique_indices)} ({len(unique_indices) / len(active_df):.1%})')

test_indeces['random'] = unique_indices

# # Get the test set
# test_df = active_df.loc[unique_indices]
# # Bar plot of the test Active distribution as percentage
# test_df['Active'].value_counts(normalize=True).plot(kind='bar')
# plt.title('Test set Active distribution')
# plt.show()
# # Bar plot of the test Active - OR distribution as percentage
# test_df['Active - OR'].value_counts(normalize=True).plot(kind='bar')
# plt.title('Test set Active - OR distribution')
# plt.show()

# Isolating the unique Uniprots:

active_df = protac_df[protac_df[active_col].notna()].copy()

unique_uniprot = active_df['Uniprot'].value_counts() == 1
print(f'Number of unique Uniprot: {unique_uniprot.sum()}')

# NOTE: Since they are very few, all unique Uniprot will be used as test set.
# Get the indices for a test set
unique_indices = active_df[active_df['Uniprot'].isin(unique_uniprot.index)].index


test_indeces['uniprot'] = unique_indices
print(f'Number of unique indices: {len(unique_indices)} ({len(unique_indices) / len(active_df):.1%})')

# DEPRECATED: The following results in a too Before starting any training, we isolate a small group of test data. Each element in the test set is selected so that all the following conditions are met:
# * its SMILES is unique
# * its POI is unique
# * its (SMILES, POI) pair is unique

active_df = protac_df[protac_df[active_col].notna()]

# Find the samples that:
# * have their SMILES appearing only once in the dataframe
# * have their Uniprot appearing only once in the dataframe
# * have their (Smiles, Uniprot) pair appearing only once in the dataframe
unique_smiles = active_df['Smiles'].value_counts() == 1
unique_uniprot = active_df['Uniprot'].value_counts() == 1
unique_smiles_uniprot = active_df.groupby(['Smiles', 'Uniprot']).size() == 1

# Get the indices of the unique samples
unique_smiles_idx = active_df['Smiles'].map(unique_smiles)
unique_uniprot_idx = active_df['Uniprot'].map(unique_uniprot)
unique_smiles_uniprot_idx = active_df.set_index(['Smiles', 'Uniprot']).index.map(unique_smiles_uniprot)

# Cross the indices to get the unique samples
# unique_samples = active_df[unique_smiles_idx & unique_uniprot_idx & unique_smiles_uniprot_idx].index
unique_samples = active_df[unique_smiles_idx & unique_uniprot_idx].index
test_df = active_df.loc[unique_samples]

warnings.filterwarnings("ignore", ".*FixedLocator*")

# ## Cross-Validation Training

# Cross validation training with 5 splits. The split operation is done in three different ways:
# 
# * Random split
# * POI-wise: some POIs never in both splits
# * Least Tanimoto similarity PROTAC-wise

# ### Plotting CV Folds 


# NOTE: When set to 60, it will result in 29 groups, with nice distributions of
# the number of unique groups in the train and validation sets, together with
# the number of active and inactive PROTACs. 
n_bins_tanimoto = 60 if active_col == 'Active' else 400
n_splits = 5
# The train and validation sets will be created from the active PROTACs only,
# i.e., the ones with 'Active' column not NaN, and that are NOT in the test set
active_df = protac_df[protac_df[active_col].notna()]
train_val_df = active_df[~active_df.index.isin(test_df.index)].copy()

# Make three groups for CV:
# * Random split
# * Split by Uniprot (POI)
# * Split by least tanimoto similarity PROTAC-wise
groups = [
    'random',
    'uniprot',
    'tanimoto',
]
for group_type in groups:
    if group_type == 'random':
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        groups = None
    elif group_type == 'uniprot':
        # Split by Uniprot
        kf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        encoder = OrdinalEncoder()
        groups = encoder.fit_transform(train_val_df['Uniprot'].values.reshape(-1, 1))
        print(f'Number of unique groups: {len(encoder.categories_[0])}')
    elif group_type == 'tanimoto':
        # Split by tanimoto similarity, i.e., group_type PROTACs with similar Avg Tanimoto
        kf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        tanimoto_groups = pd.cut(train_val_df['Avg Tanimoto'], bins=n_bins_tanimoto).copy()
        encoder = OrdinalEncoder()
        groups = encoder.fit_transform(tanimoto_groups.values.reshape(-1, 1))
        print(f'Number of unique groups: {len(encoder.categories_[0])}')
    

    X = train_val_df.drop(columns=active_col)
    y = train_val_df[active_col].tolist()

    # print(f'Group: {group_type}')
    # fig, ax = plt.subplots(figsize=(6, 3))
    # plot_cv_indices(kf, X=X, y=y, group=groups, ax=ax, n_splits=n_splits)
    # plt.tight_layout()
    # plt.show()

    stats = []
    for k, (train_index, val_index) in enumerate(kf.split(X, y, groups)):
        train_df = train_val_df.iloc[train_index]
        val_df = train_val_df.iloc[val_index]
        stat = {
            'fold': k,
            'train_len': len(train_df),
            'val_len': len(val_df),
            'train_perc': len(train_df) / len(train_val_df),
            'val_perc': len(val_df) / len(train_val_df),
            'train_active (%)': train_df[active_col].sum() / len(train_df) * 100,
            'train_inactive (%)': (len(train_df) - train_df[active_col].sum()) / len(train_df) * 100,
            'val_active (%)': val_df[active_col].sum() / len(val_df) * 100,
            'val_inactive (%)': (len(val_df) - val_df[active_col].sum()) / len(val_df) * 100,
            'num_leaking_uniprot': len(set(train_df['Uniprot']).intersection(set(val_df['Uniprot']))),
            'num_leaking_smiles': len(set(train_df['Smiles']).intersection(set(val_df['Smiles']))),
        }
        if group_type != 'random':
            stat['train_unique_groups'] = len(np.unique(groups[train_index]))
            stat['val_unique_groups'] = len(np.unique(groups[val_index]))
        stats.append(stat)
    print('-' * 120)

# ### Run CV

import warnings

# Seed everything in pytorch lightning
pl.seed_everything(42)


def train_model(
        train_df,
        val_df,
        test_df=None,
        hidden_dim=768,
        batch_size=8,
        learning_rate=2e-5,
        max_epochs=50,
        smiles_emb_dim=1024,
        smote_k_neighbors=5,
        use_ored_activity=False if active_col == 'Active' else True,
        fast_dev_run=False,
        use_logger=True,
        logger_name='protac',
        disabled_embeddings=[],
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
        use_ored_activity (bool): Whether to use the ORED activity column.
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
        use_smote=True,
        oversampler=oversampler,
        use_ored_activity=use_ored_activity,
    )
    val_ds = PROTAC_Dataset(
        val_df,
        protein_embeddings,
        cell2embedding,
        smiles2fp,
        use_ored_activity=use_ored_activity,
    )
    if test_df is not None:
        test_ds = PROTAC_Dataset(
            test_df,
            protein_embeddings,
            cell2embedding,
            smiles2fp,
            use_ored_activity=use_ored_activity,
        )
    logger = pl.loggers.TensorBoardLogger(
        save_dir='../logs',
        name=logger_name,
    )
    callbacks = [
        pl.callbacks.EarlyStopping(
            monitor='train_loss',
            patience=10,
            mode='max',
            verbose=True,
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
    )
    model = PROTAC_Model(
        hidden_dim=hidden_dim,
        smiles_emb_dim=smiles_emb_dim,
        poi_emb_dim=1024,
        e3_emb_dim=1024,
        cell_emb_dim=768,
        batch_size=batch_size,
        learning_rate=learning_rate,
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
        trial,
        train_df,
        val_df,
        hidden_dim_options,
        batch_size_options,
        learning_rate_options,
        max_epochs_options,
        smote_k_neighbors_options,
        fast_dev_run=False,
) -> float:
    # Generate the hyperparameters
    hidden_dim = trial.suggest_categorical('hidden_dim', hidden_dim_options)
    batch_size = trial.suggest_categorical('batch_size', batch_size_options)
    learning_rate = trial.suggest_float('learning_rate', *learning_rate_options, log=True)
    max_epochs = trial.suggest_categorical('max_epochs', max_epochs_options)
    smote_k_neighbors = trial.suggest_categorical('smote_k_neighbors', smote_k_neighbors_options)

    # Train the model with the current set of hyperparameters
    _, _, metrics = train_model(
        train_df,
        val_df,
        hidden_dim=hidden_dim,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        smote_k_neighbors=smote_k_neighbors,
        use_logger=False,
        fast_dev_run=fast_dev_run,
    )

    # Metrics is a dictionary containing at least the validation loss
    val_loss = metrics['val_loss']
    val_acc = metrics['val_acc']
    val_roc_auc = metrics['val_roc_auc']
    
    # Optuna aims to minimize the objective
    return val_loss - val_acc - val_roc_auc


def hyperparameter_tuning_and_training(
        train_df,
        val_df,
        test_df,
        fast_dev_run=False,
        n_trials=20,
        logger_name='protac_hparam_search',
) -> tuple:
    """ Hyperparameter tuning and training of a PROTAC model.
    
    Args:
        train_df (pd.DataFrame): The training set.
        val_df (pd.DataFrame): The validation set.
        test_df (pd.DataFrame): The test set.
        fast_dev_run (bool): Whether to run a fast development run.

    Returns:
        tuple: The trained model, the trainer, and the best metrics.
    """
    # Define the search space
    hidden_dim_options = [256, 512, 768]
    batch_size_options = [8, 16, 32]
    learning_rate_options = (1e-5, 1e-3) # min and max values for loguniform distribution
    max_epochs_options = [10, 20, 50]
    smote_k_neighbors_options = list(range(3, 16))

    # Create an Optuna study object
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(
            trial,
            train_df,
            val_df,
            hidden_dim_options,
            batch_size_options,
            learning_rate_options,
            max_epochs_options,
            smote_k_neighbors_options=smote_k_neighbors_options,
            fast_dev_run=fast_dev_run,),
        n_trials=n_trials,
    )

    # Retrieve the best hyperparameters
    best_params = study.best_params
    best_hidden_dim = best_params['hidden_dim']
    best_batch_size = best_params['batch_size']
    best_learning_rate = best_params['learning_rate']
    best_max_epochs = best_params['max_epochs']
    best_smote_k_neighbors = best_params['smote_k_neighbors']

    # Retrain the model with the best hyperparameters
    model, trainer, metrics = train_model(
        train_df,
        val_df,
        test_df,
        hidden_dim=best_hidden_dim,
        batch_size=best_batch_size,
        learning_rate=best_learning_rate,
        max_epochs=best_max_epochs,
        use_logger=True,
        logger_name=logger_name,
        fast_dev_run=fast_dev_run,
    )

    # Report the best hyperparameters found
    metrics['hidden_dim'] = best_hidden_dim
    metrics['batch_size'] = best_batch_size
    metrics['learning_rate'] = best_learning_rate
    metrics['max_epochs'] = best_max_epochs
    metrics['smote_k_neighbors'] = best_smote_k_neighbors

    # Return the best metrics
    return model, trainer, metrics

# Example usage
# train_df, val_df, test_df = load_your_data()  # You need to load your datasets here
# model, trainer, best_metrics = hyperparameter_tuning_and_training(train_df, val_df, test_df)

# Loop over the different splits and train the model:
active_name = active_col.replace(' ', '').lower()
active_name = 'active-and' if active_name == 'active' else active_name
n_splits = 5

report = []
active_df = protac_df[protac_df[active_col].notna()]
train_val_df = active_df[~active_df.index.isin(unique_samples)]

# Make directory ../reports if it does not exist
if not os.path.exists('../reports'):
    os.makedirs('../reports')

for group_type in ['random', 'uniprot', 'tanimoto']:
    print(f'Starting CV for group type: {group_type}')
    # Setup CV iterator and groups
    if group_type == 'random':
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        groups = None
    elif group_type == 'uniprot':
        # Split by Uniprot
        kf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        encoder = OrdinalEncoder()
        groups = encoder.fit_transform(train_val_df['Uniprot'].values.reshape(-1, 1))
    elif group_type == 'tanimoto':
        # Split by tanimoto similarity, i.e., group_type PROTACs with similar Avg Tanimoto
        kf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        tanimoto_groups = pd.cut(train_val_df['Avg Tanimoto'], bins=n_bins_tanimoto).copy()
        encoder = OrdinalEncoder()
        groups = encoder.fit_transform(tanimoto_groups.values.reshape(-1, 1))
    # Start the CV over the folds
    X = train_val_df.drop(columns=active_col)
    y = train_val_df[active_col].tolist()
    for k, (train_index, val_index) in enumerate(kf.split(X, y, groups)):
        train_df = train_val_df.iloc[train_index]
        val_df = train_val_df.iloc[val_index]
        stats = {
            'fold': k,
            'group_type': group_type,
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
            'num_leaking_uniprot': len(set(train_df['Uniprot']).intersection(set(val_df['Uniprot']))),
            'num_leaking_smiles': len(set(train_df['Smiles']).intersection(set(val_df['Smiles']))),
        }
        if group_type != 'random':
            stats['train_unique_groups'] = len(np.unique(groups[train_index]))
            stats['val_unique_groups'] = len(np.unique(groups[val_index]))
        # Train and evaluate the model
        # model, trainer, metrics = train_model(train_df, val_df, test_df)
        model, trainer, metrics = hyperparameter_tuning_and_training(
            train_df,
            val_df,
            test_df,
            fast_dev_run=False,
            n_trials=50,
            logger_name=f'protac_{active_name}_{group_type}_fold_{k}',
        )
        stats.update(metrics)
        del model
        del trainer
        report.append(stats)
report = pd.DataFrame(report)
report.to_csv(
    f'../reports/cv_report_hparam_search_{n_splits}-splits_{active_name}.csv',
    index=False,
)
