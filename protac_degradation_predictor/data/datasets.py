import random
import logging
from pathlib import Path
from typing import Literal, List, Tuple, Optional, Dict, Union

from torch.utils.data import Dataset
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
import numpy as np
import pandas as pd

from protac_degradation_predictor.data_utils import (
    get_fingerprint,
    is_active,
    load_cell2embedding,
    load_protein2embedding,
)
from protac_degradation_predictor.data.embeddings.mol_embeddings import (
    MolEmbedding
)
from protac_degradation_predictor.data.embeddings.protein_embeddings import (
    ProteinEmbedding
)
from protac_degradation_predictor.data.embeddings.cell_embeddings import (
    CellEmbedding
)


class PROTAC_Dataset(Dataset):

    def __init__(
        self,
        protac_df: pd.DataFrame,
        protein2embedding: Dict[str, np.ndarray],
        cell2embedding: Dict[str, np.ndarray],
        mol2embedding: Dict[str, np.ndarray],
        protac_column: str = 'PROTAC',
        poi_column: str = 'POI',
        e3_column: str = 'E3 Ligase',
        cell_column: str = 'Cell Line',
        label_column: Union[List, str] = 'Active',
        use_smote: bool = False,
        oversampler: Optional[SMOTE | ADASYN] = None,
        disabled_embeddings: List[Literal['protac', 'poi', 'e3', 'cell']] = [],
        scaler: Optional[StandardScaler | Dict[str, StandardScaler]] = None,
        use_single_scaler: Optional[bool] = None,
        shuffle_embedding_prob: float = 0.0,
    ):
        """ Initialize the PROTAC dataset.

        Args:
            protac_df (pd.DataFrame): The PROTAC dataframe
            protein2embedding (dict): Dictionary of protein embeddings
            cell2embedding (dict): Dictionary of cell line embeddings
            mol2embedding (dict): Dictionary of SMILES to fingerprint
            use_smote (bool): Whether to use SMOTE for oversampling
            oversampler (SMOTE | ADASYN): The oversampler to use
            label_column (str): The column containing the active/inactive information
            disabled_embeddings (list): The list of embeddings to disable, i.e., return a zero vector
            scaler (StandardScaler | dict): The scaler to use for the embeddings
            use_single_scaler (bool): Whether to use a single scaler for all features
            shuffle_embedding_prob (float): The probability of shuffling the embeddings. Used for testing whether embeddings act as "barcodes". Defaults to 0.0, i.e., no shuffling.
        """
        # Filter out examples with NaN in label_column column
        self.data = protac_df  # [~protac_df[label_column].isna()]
        self.protein2embedding = protein2embedding
        self.cell2embedding = cell2embedding
        self.mol2embedding = mol2embedding
        self.label_column = label_column
        self.protac_column = protac_column
        self.poi_column = poi_column
        self.e3_column = e3_column
        self.cell_column = cell_column
        self.disabled_embeddings = disabled_embeddings

        # Scaling parameters
        self.scaler = scaler
        self.use_single_scaler = use_single_scaler

        self.mol_emb_dim = mol2embedding[list(mol2embedding.keys())[0]].shape[0]
        self.protein_emb_dim = protein2embedding[list(protein2embedding.keys())[0]].shape[0]
        self.cell_emb_dim = cell2embedding[list(cell2embedding.keys())[0]].shape[0]

        self.default_smiles_emb = np.zeros(self.mol_emb_dim)
        self.default_protein_emb = np.zeros(self.protein_emb_dim)
        self.default_cell_emb = np.zeros(self.cell_emb_dim)

        # Look up the embeddings
        self.data = pd.DataFrame({
            'PROTAC': self.data['PROTAC'].apply(lambda x: mol2embedding.get(x, self.default_smiles_emb).astype(np.float32)).tolist(),
            'POI': self.data['POI'].apply(lambda x: protein2embedding.get(x, self.default_protein_emb).astype(np.float32)).tolist(),
            'E3 Ligase': self.data['E3 Ligase Uniprot'].apply(lambda x: protein2embedding.get(x, self.default_protein_emb).astype(np.float32)).tolist(),
            'Cell': self.data['Cell Line'].apply(lambda x: cell2embedding.get(x, self.default_cell_emb).astype(np.float32)).tolist(),
            self.label_column: self.data[self.label_column].astype(np.float32).tolist(),
        })

        # Apply SMOTE
        self.use_smote = use_smote
        self.oversampler = oversampler
        if self.use_smote:
            self.apply_smote()
        
        self.shuffle_embedding_prob = shuffle_embedding_prob
        if shuffle_embedding_prob > 0.0:
            # Set random seed
            random.seed(42)
            if self.protein_emb_dim != self.cell_emb_dim:
                logging.warning('Protein and cell embeddings have different dimensions. Shuffling will be on POI and E3 embeddings only.')
    
    def get_mol_emb_dim(self):
        return self.mol_emb_dim

    def get_protein_emb_dim(self):
        return self.protein_emb_dim
    
    def get_cell_emb_dim(self):
        return self.cell_emb_dim

    def apply_smote(self):
        # Prepare the dataset for SMOTE
        features = []
        labels = []
        for _, row in self.data.iterrows():
            features.append(np.hstack([
                row['PROTAC'],
                row['POI'],
                row['E3 Ligase'],
                row['Cell Line'],
            ]))
            labels.append(row[self.label_column])

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
        smiles_embs = features_smote[:, :self.mol_emb_dim]
        poi_embs = features_smote[:,
                                  self.mol_emb_dim:self.mol_emb_dim+self.protein_emb_dim]
        e3_embs = features_smote[:, self.mol_emb_dim +
                                 self.protein_emb_dim:self.mol_emb_dim+2*self.protein_emb_dim]
        cell_embs = features_smote[:, -self.cell_emb_dim:]

        # Reconstruct the dataframe with oversampled data
        df_smote = pd.DataFrame({
            'PROTAC': list(smiles_embs),
            'POI': list(poi_embs),
            'E3 Ligase': list(e3_embs),
            'Cell Line': list(cell_embs),
            self.label_column: labels_smote
        })
        self.data = df_smote

    def fit_scaling(self, use_single_scaler: bool = False, **scaler_kwargs) -> dict:
        """ Fit the scalers for the data and save them in the dataset class.

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
                np.array(self.data['PROTAC'].tolist()),
                np.array(self.data['POI'].tolist()),
                np.array(self.data['E3 Ligase'].tolist()),
                np.array(self.data['Cell Line'].tolist()),
            ])
            self.scaler.fit(embeddings)
            return self.scaler
        else:
            self.use_single_scaler = False
            scalers = {}
            scalers['PROTAC'] = StandardScaler(**scaler_kwargs)
            scalers['POI'] = StandardScaler(**scaler_kwargs)
            scalers['E3 Ligase'] = StandardScaler(**scaler_kwargs)
            scalers['Cell Line'] = StandardScaler(**scaler_kwargs)

            scalers['PROTAC'].fit(np.stack(self.data['PROTAC'].to_numpy()))
            scalers['POI'].fit(np.stack(self.data['POI'].to_numpy()))
            scalers['E3 Ligase'].fit(np.stack(self.data['E3 Ligase'].to_numpy()))
            scalers['Cell Line'].fit(np.stack(self.data['Cell Line'].to_numpy()))

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
                np.array(self.data['PROTAC'].tolist()),
                np.array(self.data['POI'].tolist()),
                np.array(self.data['E3 Ligase'].tolist()),
                np.array(self.data['Cell Line'].tolist()),
            ])
            scaled_embeddings = scalers.transform(embeddings)
            self.data = pd.DataFrame({
                'PROTAC': list(scaled_embeddings[:, :self.mol_emb_dim]),
                'POI': list(scaled_embeddings[:, self.mol_emb_dim:self.mol_emb_dim+self.protein_emb_dim]),
                'E3 Ligase': list(scaled_embeddings[:, self.mol_emb_dim+self.protein_emb_dim:self.mol_emb_dim+2*self.protein_emb_dim]),
                'Cell Line': list(scaled_embeddings[:, -self.cell_emb_dim:]),
                self.label_column: self.data[self.label_column]
            })
        else:
            # Check if the self.data[<column>] data contains only binary values
            # (0 or 1). If so, do not apply scaling.
            for feature in ['PROTAC', 'POI', 'E3 Ligase', 'Cell Line']:
                feature_array = np.array(self.data[feature].tolist())
                if np.all(np.isin(feature_array, [0, 1])):
                    continue
                self.data[feature] = self.data[feature].apply(lambda x: scalers[feature].transform(x[np.newaxis, :])[0])

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
                np.array(self.data['PROTAC'].tolist()),
                np.array(self.data['POI'].tolist()),
                np.array(self.data['E3 Ligase'].tolist()),
                np.array(self.data['Cell Line'].tolist()),
            ]).astype(np.float32).copy()
        y = self.data[self.label_column].values.copy()
        return X, y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if 'protac' in self.disabled_embeddings:
            # Get a zero vector for the fingerprint
            smiles_emb = np.zeros(self.mol_emb_dim).astype(np.float32)

            # TODO: Remove random sampling in the future
            # # Uniformly sample a binary vector for the fingerprint
            # smiles_emb = np.random.randint(0, 2, size=self.mol_emb_dim).astype(np.float32)
            # if not self.use_single_scaler and self.scaler is not None:
            #     smiles_emb = smiles_emb[np.newaxis, :]
            #     smiles_emb = self.scaler['PROTAC'].transform(smiles_emb).flatten()
        else:
            smiles_emb = self.data['PROTAC'].iloc[idx]

        if 'poi' in self.disabled_embeddings:
            poi_emb = np.zeros(self.protein_emb_dim).astype(np.float32)

            # TODO: Remove random sampling in the future
            # # Uniformly sample a vector for the protein
            # poi_emb = np.random.rand(self.protein_emb_dim).astype(np.float32)
            # if not self.use_single_scaler and self.scaler is not None:
            #     poi_emb = poi_emb[np.newaxis, :]
            #     poi_emb = self.scaler['POI'].transform(poi_emb).flatten()
        else:
            poi_emb = self.data['POI'].iloc[idx]

        if 'e3' in self.disabled_embeddings:
            e3_emb = np.zeros(self.protein_emb_dim).astype(np.float32)

            # TODO: Remove random sampling in the future
            # # Uniformly sample a vector for the E3 ligase
            # e3_emb = np.random.rand(self.protein_emb_dim).astype(np.float32)
            # if not self.use_single_scaler and self.scaler is not None:
            #     # Add extra dimension for compatibility with the scaler
            #     e3_emb = e3_emb[np.newaxis, :]
            #     e3_emb = self.scaler['E3 Ligase'].transform(e3_emb)
            #     e3_emb = e3_emb.flatten()
        else:
            e3_emb = self.data['E3 Ligase'].iloc[idx]
        
        if 'cell' in self.disabled_embeddings:
            cell_emb = np.zeros(self.cell_emb_dim).astype(np.float32)

            # TODO: Remove random sampling in the future
            # # Uniformly sample a vector for the cell line
            # cell_emb = np.random.rand(self.cell_emb_dim).astype(np.float32)
            # if not self.use_single_scaler and self.scaler is not None:
            #     cell_emb = cell_emb[np.newaxis, :]
            #     cell_emb = self.scaler['Cell Line'].transform(cell_emb).flatten()
        else:
            cell_emb = self.data['Cell Line'].iloc[idx]

        # Shuffle the embeddings if the probability is met
        if random.random() < self.shuffle_embedding_prob:
            if self.protein_emb_dim == self.cell_emb_dim:
                # Randomly shuffle the embeddings for POI, cell, and E3
                embeddings = np.vstack([poi_emb, e3_emb, cell_emb])
                np.random.shuffle(embeddings)
                poi_emb, e3_emb, cell_emb = embeddings
            else:
                # Swap POI and E3 embeddings only, because of different dimensions
                poi_emb, e3_emb = e3_emb, poi_emb

        elem = {
            'protac_emb': smiles_emb,
            'poi_emb': poi_emb,
            'e3_emb': e3_emb,
            'cell_emb': cell_emb,
            'label': self.data[self.label_column].iloc[idx],
        }
        return elem

class MolPoiE3CellDataset(Dataset):

    def __init__(
        self,
        df: pd.DataFrame,
        mol_column: str = "SMILES",
        poi_column: str = "POI",
        e3_column: str = "E3 Ligase",
        cell_column: str = "Cell Line",
        label_columns: Union[List, str] = "Active",
        use_smote: bool = False,
        oversampler: Optional[SMOTE | ADASYN] = None,
        disabled_embeddings: List[Literal["mol", "poi", "e3", "cell"]] = [],
        shuffle_embedding_prob: float = 0.0,
        
        # Embedding class initialization kwargs
        mol_embeddings_kwargs: Optional[Dict] = None,
        protein_embeddings_kwargs: Optional[Dict] = None,
        cell_embeddings_kwargs: Optional[Dict] = None,
        
        # Preprocessing options
        mol_preprocess_op: Optional[Literal["standardize", "normalize", "minmax"]] = None,
        protein_preprocess_op: Optional[Literal["standardize", "normalize", "minmax"]] = None,
        cell_preprocess_op: Optional[Literal["standardize", "normalize", "minmax"]] = None,
        imputer: Optional[Literal["simpler", "knn", "iterative"]] = None,
        imputer_kwargs: Optional[Dict[str, Union[int, str]]] = None,
        save_embeddings_to_cache: bool = True,
        label_tasks: Optional[Union[Literal["classification", "regression"], List[Literal["classification", "regression"]]]] = None,
    ):
        """ Initialize the dataset for targeted protein degradation prediction.
        
        Embeddings for molecules, proteins, and cells are loaded from the cache
        or computed at initialization. If `save_embeddings_to_cache` is True,
        the embeddings are saved to the cache after encoding.

        Args:
            df (pd.DataFrame): The dataframe to use for the dataset.
            mol_column (str): The column containing the SMILES strings.
            poi_column (str): The column containing the POI sequences.
            e3_column (str): The column containing the E3 ligase sequences.
            cell_column (str): The column containing the cell line names.
            label_columns (Union[List, str]): The columns containing the labels to predict.
            use_smote (bool): Whether to use SMOTE for oversampling. Deprecated.
            oversampler (Optional[SMOTE | ADASYN]): The oversampler to use. Deprecated.
            disabled_embeddings (List[Literal["mol", "poi", "e3", "cell"]]): The list of embeddings to disable, i.e., return a zero vector.
            shuffle_embedding_prob (float): The probability of shuffling the embeddings. Used for testing whether embeddings act as "barcodes". Defaults to 0.0, i.e., no shuffling.
            mol_embeddings_kwargs (Optional[Dict]): Keyword arguments for the MolEmbedding class initialization. Should include 'embeddings_type' and type-specific parameters.
            protein_embeddings_kwargs (Optional[Dict]): Keyword arguments for the ProteinEmbedding class initialization. Should include 'embeddings_type' and type-specific parameters.
            cell_embeddings_kwargs (Optional[Dict]): Keyword arguments for the CellEmbedding class initialization. Should include 'embeddings_type' and type-specific parameters.
            mol_preprocess_op (Optional[Literal["standardize", "normalize", "minmax"]]): Preprocessing operation to apply to the molecular embeddings. If None, no preprocessing is applied.
            protein_preprocess_op (Optional[Literal["standardize", "normalize", "minmax"]]): Preprocessing operation to apply to the protein embeddings. If None, no preprocessing is applied.
            cell_preprocess_op (Optional[Literal["standardize", "normalize", "minmax"]]): Preprocessing operation to apply to the cell embeddings. If None, no preprocessing is applied.
            imputer (str): The imputer to use for missing values in the labels. Defaults to None, i.e., no imputation. Options are 'simpler' (SimpleImputer), 'knn' (KNNImputer), and 'iterative' (IterativeImputer).
            imputer_kwargs (dict): Additional keyword arguments for the imputer. Example: {'n_neighbors': 5} for KNNImputer.
            save_embeddings_to_cache (bool): Whether to save the embeddings to cache after encoding. Defaults to True.
            label_tasks (Optional[Union[str, List[str]]]): The type of task for each label column. If a single string is provided, it will be applied to all label columns.
        """
        self.mol_column = mol_column
        self.poi_column = poi_column
        self.e3_column = e3_column
        self.cell_column = cell_column
        self.disabled_embeddings = disabled_embeddings
        self.label_columns = label_columns if isinstance(label_columns, list) else [label_columns]
        self.label_tasks = label_tasks if isinstance(label_tasks, list) else [label_tasks] * len(self.label_columns)

        # Setup embeddings with default parameters if not provided
        default_mol_kwargs = {"embeddings_type": "fingerprint", "load_from_cache": True}
        default_protein_kwargs = {"embeddings_type": "esm", "load_from_cache": True}
        default_cell_kwargs = {"embeddings_type": "sentence_transformer", "load_from_cache": True}
        
        # Merge user-provided kwargs with defaults
        mol_kwargs = {**default_mol_kwargs, **(mol_embeddings_kwargs or {})}
        protein_kwargs = {**default_protein_kwargs, **(protein_embeddings_kwargs or {})}
        cell_kwargs = {**default_cell_kwargs, **(cell_embeddings_kwargs or {})}

        self.embeddings = {
            "mol": MolEmbedding(**mol_kwargs),
            "prot": ProteinEmbedding(**protein_kwargs),
            "cell": CellEmbedding(**cell_kwargs),
        }

        # Setup internal data structure
        self.data = df[
            [mol_column, poi_column, e3_column, cell_column] + self.label_columns
        ].copy()

        # Rename columns for consistency
        usercolumn2internal = {
            mol_column: "mol",
            poi_column: "poi",
            e3_column: "e3",
            cell_column: "cell"
        }
        self.data.rename(columns=usercolumn2internal, inplace=True)

        # Map user column names to internal names, for reporting errors
        internal2usercolumn = {v: k for k, v in usercolumn2internal.items()}

        # Look up the embeddings: This will also store the embeddings in cache
        to_encode = {
            "mol": self.data["mol"],
            "prot": pd.concat([self.data["poi"], self.data["e3"]]),
            "cell": self.data["cell"],
        }
        
        for key in self.embeddings.keys():
            # Check if to_encode[key] has any NaN values
            if to_encode[key].isna().any():
                user_col = internal2usercolumn.get(key, f"{usercolumn2internal['poi']} or {usercolumn2internal['e3']}")
                raise ValueError(
                    f"NaN values found in {user_col} column(s). Please ensure all entries in input to the model are valid before encoding."
                )
            logging.debug(f"Encoding {key} embeddings...")

            # Use the simplified encode method
            self.embeddings[key].encode(
                to_encode[key].dropna().unique().tolist(),
                skip_existing=True,
                update_cache=save_embeddings_to_cache,
            )

        # Apply preprocessing operations
        # NOTE: The preprocessed embeddings are NOT saved to cache
        if mol_preprocess_op:
            logging.debug("Preprocessing molecular embeddings...")
            scaled_embeddings = self.embeddings["mol"].preprocess(mol_preprocess_op)
            self.embeddings["mol"].update(scaled_embeddings)

        if protein_preprocess_op:
            logging.debug("Preprocessing protein embeddings...")
            scaled_embeddings = self.embeddings["prot"].preprocess(protein_preprocess_op)
            self.embeddings["prot"].update(scaled_embeddings)

        if cell_preprocess_op:
            logging.debug("Preprocessing cell embeddings...")
            scaled_embeddings = self.embeddings["cell"].preprocess(cell_preprocess_op)
            self.embeddings["cell"].update(scaled_embeddings)

        # Handle missing values in the labels with imputation
        y = self.data[self.label_columns].values.astype(np.float32)

        if np.isnan(y).any() and imputer is not None:
            logging.debug("Missing values found in the labels. Applying imputation...")
            if imputer == "simpler":
                imputer_obj = SimpleImputer(strategy="mean")
            elif imputer == "knn":
                imputer_obj = KNNImputer(**(imputer_kwargs or {}))
            elif imputer == "iterative":
                imputer_obj = IterativeImputer(**(imputer_kwargs or {}))
            else:
                raise ValueError(f"Invalid imputer: {imputer}. Choose from 'simpler', 'knn', or 'iterative'.")
            y = imputer_obj.fit_transform(y)

            logging.debug("Imputing missing values in the labels...")
            # Update the data with the imputed labels
            if isinstance(self.label_columns, str):
                self.data[self.label_columns] = pd.Series(
                    y.flatten(), index=self.data.index
                )
            else:
                # If label_columns is a list, we need to create a DataFrame
                # with the same index as self.data
                self.data[self.label_columns] = pd.DataFrame(
                    y, columns=self.label_columns, index=self.data.index
                )

        # Get the dimensions of the embeddings
        # NOTE: This will only work if the embeddings are already loaded
        logging.debug("Getting the dimensions of the molecular embeddings...")
        self.mol_emb_dim = self.embeddings["mol"].shape()
        logging.debug("Getting the dimensions of the protein embeddings...")
        self.prot_emb_dim = self.embeddings["prot"].shape()
        logging.debug("Getting the dimensions of the cell embeddings...")
        self.cell_emb_dim = self.embeddings["cell"].shape()
        logging.debug(f"Mol emb dim: {self.mol_emb_dim}, Prot emb dim: {self.prot_emb_dim}, Cell emb dim: {self.cell_emb_dim}")

        # Apply SMOTE
        self.use_smote = use_smote
        self.oversampler = oversampler
        if False and self.use_smote:
            # TODO: Implement SMOTE, only applicable to classification tasks!
            self.apply_smote()
        
        self.shuffle_embedding_prob = shuffle_embedding_prob
        if shuffle_embedding_prob > 0.0:
            # Set random seed
            random.seed(42)
            if self.prot_emb_dim != self.cell_emb_dim:
                logging.warning('Protein and cell embeddings have different dimensions. Shuffling will be on POI and E3 embeddings only.')

    def apply_smote(self):
        """Apply SMOTE oversampling in embedding space for classification.

        Notes:
        - Only applicable when all tasks are classification and a single
          label column is provided. For multi-label/mixed tasks, raises
          a ValueError.
        - Operates in continuous embedding space by constructing X from
          cached embeddings, then storing resampled arrays internally.
        """
        # Validate task type and label dimensionality
        if any(task != "classification" for task in self.label_tasks):
            raise ValueError("SMOTE is only applicable to classification tasks.")

        if len(self.label_columns) != 1:
            raise ValueError(
                "SMOTE currently supports a single classification label column."
            )

        # Build feature matrix X by stacking embeddings for mol, poi, e3, cell
        mol_vecs = np.array([self.embeddings["mol"][x] for x in self.data["mol"].tolist()])
        poi_vecs = np.array([self.embeddings["prot"][x] for x in self.data["poi"].tolist()])
        e3_vecs = np.array([self.embeddings["prot"][x] for x in self.data["e3"].tolist()])
        cell_vecs = np.array([self.embeddings["cell"][x] for x in self.data["cell"].tolist()])

        # Ensure 2D arrays
        if mol_vecs.ndim == 1:
            mol_vecs = mol_vecs.reshape(1, -1)
        if poi_vecs.ndim == 1:
            poi_vecs = poi_vecs.reshape(1, -1)
        if e3_vecs.ndim == 1:
            e3_vecs = e3_vecs.reshape(1, -1)
        if cell_vecs.ndim == 1:
            cell_vecs = cell_vecs.reshape(1, -1)

        X = np.hstack([mol_vecs, poi_vecs, e3_vecs, cell_vecs]).astype(np.float32)
        y = self.data[self.label_columns].astype(np.float32).values.ravel()

        # Initialize SMOTE and fit
        oversampler = self.oversampler if self.oversampler is not None else SMOTE(random_state=42)
        X_res, y_res = oversampler.fit_resample(X, y)

        # Keep resampled arrays internally and mark active
        self._smote_X = X_res.astype(np.float32)
        self._smote_y = y_res.reshape(-1, 1).astype(np.float32)
        self._smote_applied = True

        # Cache split indices for quick slicing in __getitem__/to_numpy
        get_dim = lambda d: d[-1] if isinstance(d, tuple) else int(d)
        mol_dim = get_dim(self.mol_emb_dim)
        prot_dim = get_dim(self.prot_emb_dim)
        cell_dim = get_dim(self.cell_emb_dim)
        self._smote_slices = (
            slice(0, mol_dim),
            slice(mol_dim, mol_dim + prot_dim),
            slice(mol_dim + prot_dim, mol_dim + 2 * prot_dim),
            slice(mol_dim + 2 * prot_dim, mol_dim + 2 * prot_dim + cell_dim),
        )

    def to_numpy(
            self,
            component: Optional[Literal["mol", "poi", "e3", "cell"]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """ Get the numpy arrays for the dataset as features X and labels y.

        Args:
            component (str): The specific input component to get as numpy array for X. Defaults to None, i.e., get a single stacked array of all input components (i.e., molecule, POI, E3 ligase, and cell line).

        Returns:
            tuple: The numpy arrays for the dataset. The first element is the input array X, and the second element is the output array y.
        """
        # If SMOTE was applied, return resampled arrays
        if hasattr(self, "_smote_applied") and getattr(self, "_smote_applied", False):
            if component is None:
                return self._smote_X, self._smote_y

            # Map component to slice
            s_mol, s_poi, s_e3, s_cell = self._smote_slices
            comp2slice = {"mol": s_mol, "poi": s_poi, "e3": s_e3, "cell": s_cell}
            if component not in comp2slice:
                raise ValueError(f"Invalid component: {component}. Choose from 'mol', 'poi', 'e3', or 'cell'.")
            return self._smote_X[:, comp2slice[component]], self._smote_y

        # Default behavior without SMOTE
        if component is not None:
            if component == "mol":
                embeddings = self.embeddings["mol"]
            elif component == "poi" or component == "e3":
                embeddings = self.embeddings["prot"]
            elif component == "cell":
                embeddings = self.embeddings["cell"]
            else:
                raise ValueError(f"Invalid component: {component}. Choose from 'mol', 'poi', 'e3', or 'cell'.")
            X = self.data[component].tolist()
            X = np.array([embeddings[x] for x in X])
        else:
            X = np.hstack([
                np.array([self.embeddings["mol"][x] for x in self.data["mol"].tolist()]),
                np.array([self.embeddings["prot"][x] for x in self.data["poi"].tolist()]),
                np.array([self.embeddings["prot"][x] for x in self.data["e3"].tolist()]),
                np.array([self.embeddings["cell"][x] for x in self.data["cell"].tolist()]),
            ]).astype(np.float32)

        y = self.data[self.label_columns].astype(np.float32).values

        return X, y

    def shapes(self) -> Dict[str, int]:
        """ Get the dimensions of the embeddings.

        Returns:
            dict: A dictionary with the dimensions of each embedding.
        """
        get_dim = lambda d: d[-1] if isinstance(d, tuple) else int(d)
        return {
            "mol": get_dim(self.mol_emb_dim),
            "poi": get_dim(self.prot_emb_dim),
            "e3": get_dim(self.prot_emb_dim),
            "cell": get_dim(self.cell_emb_dim),
        }

    def __len__(self):
        if hasattr(self, "_smote_applied") and getattr(self, "_smote_applied", False):
            return len(self._smote_y)
        return len(self.data)

    def __getitem__(self, idx):
        # If SMOTE applied, use resampled arrays directly
        if False and hasattr(self, "_smote_applied") and getattr(self, "_smote_applied", False):
            # TODO: Double check the correcteness of this path
            s_mol, s_poi, s_e3, s_cell = self._smote_slices
            x = self._smote_X[idx]

            elem = {
                "mol": np.zeros((s_mol.stop - s_mol.start,), dtype=np.float32),
                "poi": np.zeros((s_poi.stop - s_poi.start,), dtype=np.float32),
                "e3": np.zeros((s_e3.stop - s_e3.start,), dtype=np.float32),
                "cell": np.zeros((s_cell.stop - s_cell.start,), dtype=np.float32),
            }
            if "mol" not in self.disabled_embeddings:
                elem["mol"] = x[s_mol].astype(np.float32)
            if "poi" not in self.disabled_embeddings:
                elem["poi"] = x[s_poi].astype(np.float32)
            if "e3" not in self.disabled_embeddings:
                elem["e3"] = x[s_e3].astype(np.float32)
            if "cell" not in self.disabled_embeddings:
                elem["cell"] = x[s_cell].astype(np.float32)

            # Add labels (single classification column by construction)
            for label, task in zip(self.label_columns, self.label_tasks):
                val = self._smote_y[idx, 0]
                elem[label] = np.array(val, dtype=np.float32) if task == "regression" else float(val)
            return elem

        # Default path without SMOTE
        # Default element structure to zero vectors
        elem = {
            "mol": np.zeros(self.embeddings["mol"].shape()),
            "poi": np.zeros(self.embeddings["prot"].shape()),
            "e3": np.zeros(self.embeddings["prot"].shape()),
            "cell": np.zeros(self.embeddings["cell"].shape()),
        }

        # Get the embeddings for the current index if not disabled
        if "mol" not in self.disabled_embeddings:
            elem["mol"] = self.embeddings["mol"][self.data["mol"].iloc[idx]]

        if "poi" not in self.disabled_embeddings:
            elem["poi"] = self.embeddings["prot"][self.data["poi"].iloc[idx]]
        
        if "e3" not in self.disabled_embeddings:
            elem["e3"] = self.embeddings["prot"][self.data["e3"].iloc[idx]]
        
        if "cell" not in self.disabled_embeddings:
            elem["cell"] = self.embeddings["cell"][self.data["cell"].iloc[idx]]

        # Convert all embeddings to float32
        elem = {k: v.astype(np.float32) for k, v in elem.items()}
        
        # Update the element with the labels
        for label, task in zip(self.label_columns, self.label_tasks):
            elem[label] = self.data[label].iloc[idx]
            if task == "regression":
                elem[label] = np.array(elem[label], dtype=np.float32)

        return elem
