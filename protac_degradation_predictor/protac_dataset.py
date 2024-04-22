from typing import Literal, List, Tuple, Optional, Dict

from torch.utils.data import Dataset
import numpy as np
from imblearn.over_sampling import SMOTE, ADASYN
import pandas as pd
from sklearn.preprocessing import StandardScaler


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
        # Filter out examples with NaN in active_col column
        self.data = protac_df  # [~protac_df[active_col].isna()]
        self.protein2embedding = protein2embedding
        self.cell2embedding = cell2embedding
        self.smiles2fp = smiles2fp
        self.active_label = active_label
        self.use_single_scaler = None

        self.smiles_emb_dim = smiles2fp[list(smiles2fp.keys())[0]].shape[0]
        self.protein_emb_dim = protein2embedding[list(
            protein2embedding.keys())[0]].shape[0]
        self.cell_emb_dim = cell2embedding[list(
            cell2embedding.keys())[0]].shape[0]

        # Look up the embeddings
        self.data = pd.DataFrame({
            'Smiles': self.data['Smiles'].apply(lambda x: smiles2fp[x].astype(np.float32)).tolist(),
            'Uniprot': self.data['Uniprot'].apply(lambda x: protein2embedding[x].astype(np.float32)).tolist(),
            'E3 Ligase Uniprot': self.data['E3 Ligase Uniprot'].apply(lambda x: protein2embedding[x].astype(np.float32)).tolist(),
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

            return scalers

    def apply_scaling(self, scalers: dict, use_single_scaler: bool = False):
        """ Apply scaling to the data.

        Args:
            scalers (dict): The scalers for each feature.
            use_single_scaler (bool): Whether to use a single scaler for all features.
        """
        if self.use_single_scaler is None:
            raise ValueError(
                "The fit_scaling method must be called before apply_scaling.")
        if use_single_scaler != self.use_single_scaler:
            raise ValueError(
                f"The use_single_scaler parameter must be the same as the one used in the fit_scaling method. Got {use_single_scaler}, previously {self.use_single_scaler}.")
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

    def get_numpy_arrays(self):
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
        elem = {
            'smiles_emb': self.data['Smiles'].iloc[idx],
            'poi_emb': self.data['Uniprot'].iloc[idx],
            'e3_emb': self.data['E3 Ligase Uniprot'].iloc[idx],
            'cell_emb': self.data['Cell Line Identifier'].iloc[idx],
            'active': self.data[self.active_label].iloc[idx],
        }
        return elem