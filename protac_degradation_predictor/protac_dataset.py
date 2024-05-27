from typing import Literal, List, Tuple, Optional, Dict
from collections import defaultdict

from .data_utils import (
    get_fingerprint,
    is_active,
    load_cell2embedding,
    load_protein2embedding,
)

from torch.utils.data import Dataset, DataLoader
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs


class PROTAC_Dataset(Dataset):

    def __init__(
        self,
        protac_df: pd.DataFrame,
        protein2embedding: Dict,
        cell2embedding: Dict,
        smiles2fp: Dict,
        use_smote: bool = False,
        oversampler: Optional[SMOTE | ADASYN] = None,
        active_label: str = 'Active',
        disabled_embeddings: List[Literal['smiles', 'poi', 'e3', 'cell']] = [],
        scaler: Optional[StandardScaler | Dict[str, StandardScaler]] = None,
        use_single_scaler: Optional[bool] = None,
    ):
        """ Initialize the PROTAC dataset

        Args:
            protac_df (pd.DataFrame): The PROTAC dataframe
            protein2embedding (dict): Dictionary of protein embeddings
            cell2embedding (dict): Dictionary of cell line embeddings
            smiles2fp (dict): Dictionary of SMILES to fingerprint
            use_smote (bool): Whether to use SMOTE for oversampling
            use_ored_activity (bool): Whether to use the 'Active - OR' column
        """
        # Filter out examples with NaN in active_label column
        self.data = protac_df  # [~protac_df[active_label].isna()]
        self.protein2embedding = protein2embedding
        self.cell2embedding = cell2embedding
        self.smiles2fp = smiles2fp
        self.active_label = active_label
        self.disabled_embeddings = disabled_embeddings

        # Scaling parameters
        self.scaler = scaler
        self.use_single_scaler = use_single_scaler

        self.smiles_emb_dim = smiles2fp[list(smiles2fp.keys())[0]].shape[0]
        self.protein_emb_dim = protein2embedding[list(
            protein2embedding.keys())[0]].shape[0]
        self.cell_emb_dim = cell2embedding[list(
            cell2embedding.keys())[0]].shape[0]
        
        self.default_smiles_emb = np.zeros(self.smiles_emb_dim)
        self.default_protein_emb = np.zeros(self.protein_emb_dim)
        self.default_cell_emb = np.zeros(self.cell_emb_dim)

        # Look up the embeddings
        self.data = pd.DataFrame({
            'Smiles': self.data['Smiles'].apply(lambda x: smiles2fp.get(x, self.default_smiles_emb).astype(np.float32)).tolist(),
            'Uniprot': self.data['Uniprot'].apply(lambda x: protein2embedding.get(x, self.default_protein_emb).astype(np.float32)).tolist(),
            'E3 Ligase Uniprot': self.data['E3 Ligase Uniprot'].apply(lambda x: protein2embedding.get(x, self.default_protein_emb).astype(np.float32)).tolist(),
            'Cell Line Identifier': self.data['Cell Line Identifier'].apply(lambda x: cell2embedding.get(x, self.default_cell_emb).astype(np.float32)).tolist(),
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

    def fit_scaling(self, use_single_scaler: bool = False, **scaler_kwargs) -> dict:
        """ Fit the scalers for the data.

        Args:
            use_single_scaler (bool): Whether to use a single scaler for all features.
            scaler_kwargs: Keyword arguments for the StandardScaler.

        Returns:
            dict: The fitted scalers.
        """
        if use_single_scaler:
            self.use_single_scaler = True
            self.scaler = StandardScaler(**scaler_kwargs)
            embeddings = np.hstack([
                np.array(self.data['Smiles'].tolist()),
                np.array(self.data['Uniprot'].tolist()),
                np.array(self.data['E3 Ligase Uniprot'].tolist()),
                np.array(self.data['Cell Line Identifier'].tolist()),
            ])
            self.scaler.fit(embeddings)
            return self.scaler
        else:
            self.use_single_scaler = False
            scalers = {}
            scalers['Smiles'] = StandardScaler(**scaler_kwargs)
            scalers['Uniprot'] = StandardScaler(**scaler_kwargs)
            scalers['E3 Ligase Uniprot'] = StandardScaler(**scaler_kwargs)
            scalers['Cell Line Identifier'] = StandardScaler(**scaler_kwargs)

            scalers['Smiles'].fit(np.stack(self.data['Smiles'].to_numpy()))
            scalers['Uniprot'].fit(np.stack(self.data['Uniprot'].to_numpy()))
            scalers['E3 Ligase Uniprot'].fit(np.stack(self.data['E3 Ligase Uniprot'].to_numpy()))
            scalers['Cell Line Identifier'].fit(np.stack(self.data['Cell Line Identifier'].to_numpy()))

            self.scaler = scalers
            return scalers

    def apply_scaling(self, scalers: dict, use_single_scaler: bool = False):
        """ Apply scaling to the data.

        Args:
            scalers (dict): The scalers for each feature.
            use_single_scaler (bool): Whether to use a single scaler for all features.
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
            # NOTE: The fingerprints are already in [0, 1], no need to scale them
            # self.data['Smiles'] = self.data['Smiles'].apply(lambda x: scalers['Smiles'].transform(x[np.newaxis, :])[0])
            self.data['Uniprot'] = self.data['Uniprot'].apply(lambda x: scalers['Uniprot'].transform(x[np.newaxis, :])[0])
            self.data['E3 Ligase Uniprot'] = self.data['E3 Ligase Uniprot'].apply(lambda x: scalers['E3 Ligase Uniprot'].transform(x[np.newaxis, :])[0])
            self.data['Cell Line Identifier'] = self.data['Cell Line Identifier'].apply(lambda x: scalers['Cell Line Identifier'].transform(x[np.newaxis, :])[0])

    def get_numpy_arrays(self, component: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """ Get the numpy arrays for the dataset.

        Args:
            component (str): The component to get the numpy arrays for. Defaults to None, i.e., get a single stacked array.
        
        Returns:
            tuple: The numpy arrays for the dataset. The first element is the input array, and the second element is the output array.
        """
        if component is not None:
            X = np.array(self.data[component].tolist()).copy()
        else:
            X = np.hstack([
                np.array(self.data['Smiles'].tolist()),
                np.array(self.data['Uniprot'].tolist()),
                np.array(self.data['E3 Ligase Uniprot'].tolist()),
                np.array(self.data['Cell Line Identifier'].tolist()),
            ]).copy()
        y = self.data[self.active_label].values.copy()
        return X, y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if 'smiles' in self.disabled_embeddings:
            # Get a zero vector for the fingerprint
            smiles_emb = np.zeros(self.smiles_emb_dim).astype(np.float32)

            # TODO: Remove random sampling in the future
            # # Uniformly sample a binary vector for the fingerprint
            # smiles_emb = np.random.randint(0, 2, size=self.smiles_emb_dim).astype(np.float32)
            # if not self.use_single_scaler and self.scaler is not None:
            #     smiles_emb = smiles_emb[np.newaxis, :]
            #     smiles_emb = self.scaler['Smiles'].transform(smiles_emb).flatten()
        else:
            smiles_emb = self.data['Smiles'].iloc[idx]

        if 'poi' in self.disabled_embeddings:
            poi_emb = np.zeros(self.protein_emb_dim).astype(np.float32)

            # TODO: Remove random sampling in the future
            # # Uniformly sample a vector for the protein
            # poi_emb = np.random.rand(self.protein_emb_dim).astype(np.float32)
            # if not self.use_single_scaler and self.scaler is not None:
            #     poi_emb = poi_emb[np.newaxis, :]
            #     poi_emb = self.scaler['Uniprot'].transform(poi_emb).flatten()
        else:
            poi_emb = self.data['Uniprot'].iloc[idx]

        if 'e3' in self.disabled_embeddings:
            e3_emb = np.zeros(self.protein_emb_dim).astype(np.float32)

            # TODO: Remove random sampling in the future
            # # Uniformly sample a vector for the E3 ligase
            # e3_emb = np.random.rand(self.protein_emb_dim).astype(np.float32)
            # if not self.use_single_scaler and self.scaler is not None:
            #     # Add extra dimension for compatibility with the scaler
            #     e3_emb = e3_emb[np.newaxis, :]
            #     e3_emb = self.scaler['E3 Ligase Uniprot'].transform(e3_emb)
            #     e3_emb = e3_emb.flatten()
        else:
            e3_emb = self.data['E3 Ligase Uniprot'].iloc[idx]
        
        if 'cell' in self.disabled_embeddings:
            cell_emb = np.zeros(self.cell_emb_dim).astype(np.float32)

            # TODO: Remove random sampling in the future
            # # Uniformly sample a vector for the cell line
            # cell_emb = np.random.rand(self.cell_emb_dim).astype(np.float32)
            # if not self.use_single_scaler and self.scaler is not None:
            #     cell_emb = cell_emb[np.newaxis, :]
            #     cell_emb = self.scaler['Cell Line Identifier'].transform(cell_emb).flatten()
        else:
            cell_emb = self.data['Cell Line Identifier'].iloc[idx]

        elem = {
            'smiles_emb': smiles_emb,
            'poi_emb': poi_emb,
            'e3_emb': e3_emb,
            'cell_emb': cell_emb,
            'active': self.data[self.active_label].iloc[idx],
        }
        return elem


def get_datasets(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: Optional[pd.DataFrame] = None,
        protein2embedding: Dict = None,
        cell2embedding: Dict = None,
        smiles2fp: Dict = None,
        use_smote: bool = True,
        smote_k_neighbors: int = 5,
        active_label: str = 'Active',
        disabled_embeddings: List[Literal['smiles', 'poi', 'e3', 'cell']] = [],
        scaler: Optional[StandardScaler | Dict[str, StandardScaler]] = None,
        use_single_scaler: Optional[bool] = None,
) -> Tuple[PROTAC_Dataset, PROTAC_Dataset, Optional[PROTAC_Dataset]]:
    """ Get the datasets for training the PROTAC model. """
    oversampler = SMOTE(k_neighbors=smote_k_neighbors, random_state=42)
    train_ds = PROTAC_Dataset(
        train_df,
        protein2embedding,
        cell2embedding,
        smiles2fp,
        use_smote=use_smote,
        oversampler=oversampler if use_smote else None,
        active_label=active_label,
        disabled_embeddings=disabled_embeddings,
        scaler=scaler,
        use_single_scaler=use_single_scaler,
    )
    val_ds = PROTAC_Dataset(
        val_df,
        protein2embedding,
        cell2embedding,
        smiles2fp,
        active_label=active_label,
        disabled_embeddings=disabled_embeddings,
        scaler=train_ds.scaler if train_ds.scaler is not None else scaler,
        use_single_scaler=train_ds.use_single_scaler if train_ds.use_single_scaler is not None else use_single_scaler,
    )
    if test_df is not None:
        test_ds = PROTAC_Dataset(
            test_df,
            protein2embedding,
            cell2embedding,
            smiles2fp,
            active_label=active_label,
            disabled_embeddings=disabled_embeddings,
            scaler=train_ds.scaler if train_ds.scaler is not None else scaler,
            use_single_scaler=train_ds.use_single_scaler if train_ds.use_single_scaler is not None else use_single_scaler,
        )
    else:
        test_ds = None
    return train_ds, val_ds, test_ds


class PROTAC_DataModule(pl.LightningDataModule):
    """ PyTorch Lightning DataModule for the PROTAC dataset.

    TODO: Work in progress. It would be nice to wrap all information into a
    single class, but it is not clear how to do it yet due to cross-validation
    and the need to split the data into training, validation, and test sets
    accordingly.
    
    Args:
        protac_csv_filepath (str): The path to the PROTAC CSV file.
        protein2embedding_filepath (str): The path to the protein to embedding dictionary.
        cell2embedding_filepath (str): The path to the cell line to embedding dictionary.
        pDC50_threshold (float): The threshold for the pDC50 value to consider a PROTAC active.
        Dmax_threshold (float): The threshold for the Dmax value to consider a PROTAC active.
        use_smote (bool): Whether to use SMOTE for oversampling.
        smote_k_neighbors (int): The number of neighbors to use for SMOTE.
        active_label (str): The column containing the active/inactive information.
        disabled_embeddings (list): The list of embeddings to disable.
        scaler (StandardScaler | dict): The scaler to use for the embeddings.
        use_single_scaler (bool): Whether to use a single scaler for all features.
    """

    def __init__(
        self,
        protac_csv_filepath: str,
        protein2embedding_filepath: str,
        cell2embedding_filepath: str,
        pDC50_threshold: float = 6.0,
        Dmax_threshold: float = 0.6,
        use_smote: bool = True,
        smote_k_neighbors: int = 5,
        active_label: str = 'Active',
        disabled_embeddings: List[Literal['smiles', 'poi', 'e3', 'cell']] = [],
        scaler: Optional[StandardScaler | Dict[str, StandardScaler]] = None,
        use_single_scaler: Optional[bool] = None,
    ):
        super(PROTAC_DataModule, self).__init__()

        # Load the PROTAC dataset
        self.protac_df = pd.read_csv('../data/PROTAC-Degradation-DB.csv')
        # Map E3 Ligase Iap to IAP
        self.protac_df['E3 Ligase'] = self.protac_df['E3 Ligase'].str.replace('Iap', 'IAP')
        self.protac_df[active_label] = self.protac_df.apply(
            lambda x: is_active(
                x['DC50 (nM)'],
                x['Dmax (%)'],
                pDC50_threshold=pDC50_threshold,
                Dmax_threshold=Dmax_threshold,
            ),
            axis=1,
        )
        self.smiles2fp, self.protac_df = self.get_smiles2fp_and_avg_tanimoto(self.protac_df)
        self.active_df = self.protac_df[self.protac_df[active_label].notna()].copy()

        # Load embedding dictionaries
        self.protein2embedding = load_protein2embedding(protein2embedding_filepath)
        self.cell2embedding = load_cell2embedding(cell2embedding_filepath)


    def setup(self, stage: str):
        self.train_ds, self.val_ds, self.test_ds = get_datasets(
            self.train_df,
            self.val_df,
            self.test_df,
            self.protein2embedding,
            self.cell2embedding,
            self.smiles2fp,
            use_smote=self.use_smote,
            smote_k_neighbors=self.smote_k_neighbors,
            active_label=self.active_label,
            disabled_embeddings=self.disabled_embeddings,
            scaler=self.scaler,
            use_single_scaler=self.use_single_scaler,
        )

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=32, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=32)
    
    @staticmethod
    def get_random_split_indices(active_df: pd.DataFrame, test_split: float) -> pd.Index:
        """ Get the indices of the test set using a random split.
        
        Args:
            active_df (pd.DataFrame): The DataFrame containing the active PROTACs.
            test_split (float): The percentage of the active PROTACs to use as the test set.
        
        Returns:
            pd.Index: The indices of the test set.
        """
        return active_df.sample(frac=test_split, random_state=42).index

    @staticmethod
    def get_e3_ligase_split_indices(active_df: pd.DataFrame) -> pd.Index:
        """ Get the indices of the test set using the E3 ligase split.
        
        Args:
            active_df (pd.DataFrame): The DataFrame containing the active PROTACs.
        
        Returns:
            pd.Index: The indices of the test set.
        """
        encoder = OrdinalEncoder()
        active_df['E3 Group'] = encoder.fit_transform(active_df[['E3 Ligase']]).astype(int)
        test_df = active_df[(active_df['E3 Ligase'] != 'VHL') & (active_df['E3 Ligase'] != 'CRBN')]
        return test_df.index

    @staticmethod
    def get_smiles2fp_and_avg_tanimoto(protac_df: pd.DataFrame) -> tuple:
        """ Get the SMILES to fingerprint dictionary and the average Tanimoto similarity.
        
        Args:
            protac_df (pd.DataFrame): The DataFrame containing the PROTACs.
        
        Returns:
            tuple: The SMILES to fingerprint dictionary and the average Tanimoto similarity.
        """
        unique_smiles = protac_df['Smiles'].unique().tolist()

        smiles2fp = {}
        for smiles in unique_smiles:
            smiles2fp[smiles] = get_fingerprint(smiles)

        tanimoto_matrix = defaultdict(list)
        fps = list(smiles2fp.values())

        # Compute all-against-all Tanimoto similarity using BulkTanimotoSimilarity
        for i, (smiles1, fp1) in enumerate(zip(unique_smiles, fps)):
            similarities = DataStructs.BulkTanimotoSimilarity(fp1, fps[i:])  # Only compute for i to end, avoiding duplicates
            for j, similarity in enumerate(similarities):
                distance = 1 - similarity
                tanimoto_matrix[smiles1].append(distance)  # Store as distance
                if i != i + j:
                    tanimoto_matrix[unique_smiles[i + j]].append(distance)  # Symmetric filling

        # Calculate average Tanimoto distance for each unique SMILES
        avg_tanimoto = {k: np.mean(v) for k, v in tanimoto_matrix.items()}
        protac_df['Avg Tanimoto'] = protac_df['Smiles'].map(avg_tanimoto)

        smiles2fp = {s: np.array(fp) for s, fp in smiles2fp.items()}

        return smiles2fp, protac_df

    @staticmethod
    def get_tanimoto_split_indices(
            active_df: pd.DataFrame,
            active_label: str,
            test_split: float,
            n_bins_tanimoto: int = 200,
    ) -> pd.Index:
        """ Get the indices of the test set using the Tanimoto-based split.
        
        Args:
            active_df (pd.DataFrame): The DataFrame containing the active PROTACs.
            n_bins_tanimoto (int): The number of bins to use for the Tanimoto similarity.
        
        Returns:
            pd.Index: The indices of the test set.
        """
        tanimoto_groups = pd.cut(active_df['Avg Tanimoto'], bins=n_bins_tanimoto).copy()
        encoder = OrdinalEncoder()
        active_df['Tanimoto Group'] = encoder.fit_transform(tanimoto_groups.values.reshape(-1, 1)).astype(int)
        # Sort the groups so that samples with the highest tanimoto similarity,
        # i.e., the "less similar" ones, are placed in the test set first
        tanimoto_groups = active_df.groupby('Tanimoto Group')['Avg Tanimoto'].mean().sort_values(ascending=False).index

        test_df = []
        # For each group, get the number of active and inactive entries. Then, add those
        # entries to the test_df if: 1) the test_df lenght + the group entries is less
        # 20% of the active_df lenght, and 2) the percentage of True and False entries
        # in the active_label in test_df is roughly 50%.
        for group in tanimoto_groups:
            group_df = active_df[active_df['Tanimoto Group'] == group]
            if test_df == []:
                test_df.append(group_df)
                continue
            
            num_entries = len(group_df)
            num_active_group = group_df[active_label].sum()
            num_inactive_group = num_entries - num_active_group

            tmp_test_df = pd.concat(test_df)
            num_entries_test = len(tmp_test_df)
            num_active_test = tmp_test_df[active_label].sum()
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
        return test_df.index

    @staticmethod
    def get_target_split_indices(active_df: pd.DataFrame, active_label: str, test_split: float) -> pd.Index:
        """ Get the indices of the test set using the target-based split.

        Args:
            active_df (pd.DataFrame): The DataFrame containing the active PROTACs.
            active_label (str): The column containing the active/inactive information.
            test_split (float): The percentage of the active PROTACs to use as the test set.

        Returns:
            pd.Index: The indices of the test set.
        """
        encoder = OrdinalEncoder()
        active_df['Uniprot Group'] = encoder.fit_transform(active_df[['Uniprot']]).astype(int)

        test_df = []
        # For each group, get the number of active and inactive entries. Then, add those
        # entries to the test_df if: 1) the test_df lenght + the group entries is less
        # 20% of the active_df lenght, and 2) the percentage of True and False entries
        # in the active_label in test_df is roughly 50%.
        # Start the loop from the groups containing the smallest number of entries.
        for group in reversed(active_df['Uniprot'].value_counts().index):
            group_df = active_df[active_df['Uniprot'] == group]
            if test_df == []:
                test_df.append(group_df)
                continue
            
            num_entries = len(group_df)
            num_active_group = group_df[active_label].sum()
            num_inactive_group = num_entries - num_active_group

            tmp_test_df = pd.concat(test_df)
            num_entries_test = len(tmp_test_df)
            num_active_test = tmp_test_df[active_label].sum()
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
        return test_df.index