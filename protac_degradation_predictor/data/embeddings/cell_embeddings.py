""" Handles cell embeddings using SciKit-Learn and Hugging Face Transformers. """
import re
from pathlib import Path
from typing import Optional, List, Union, Literal, Dict, Tuple
import requests
import logging

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import OneHotEncoder
from sentence_transformers import SentenceTransformer
from thefuzz import process

from protac_degradation_predictor.data.utils import get_cache_dir
from protac_degradation_predictor.data.embeddings.utils import EmbeddingMixin


class CellEmbedding(EmbeddingMixin):
    """ Class for handling cell line embeddings. """

    def __init__(
        self,
        embeddings_type: Literal["one_hot", "transformer", "sentence_transformer"] = "sentence_transformer",
        
        # One-hot encoding configurations
        onehot_enc_kwargs: Optional[dict] = None,
        
        # Transformer configurations
        pretrained_model: str = "sentence-transformers/all-mpnet-base-v1",
        batch_size: int = 64,
        device: Union[int, str] = "cpu",
        pooling: Literal["cls", "mean", "sum", "max", "mean_sqrt_len"] = "sum",
        return_tensors: Literal["pt", "np"] = "np",
        
        # Cell line processing configurations
        get_cellosaurus_descriptions: bool = True,
        min_similarity_score: float = 90,
        
        # EmbeddingMixin parameters
        embeddings: Optional[Union[Dict[str, np.ndarray], np.ndarray]] = None,
        model: Optional[Union[AutoModel, str]] = None,
        tokenizer: Optional[Union[AutoTokenizer, str]] = None,
        load_from_cache: bool = False,
        filename: Optional[Union[Path, str]] = None,
        cache_dir: Optional[Union[Path, str]] = None,
    ):
        """ Initialize the CellEmbedding class.
        
        Args:
            embeddings_type: Type of embeddings to compute consistently for this instance
            onehot_enc_kwargs: Parameters for OneHotEncoder (only used if embeddings_type="one_hot")
            pretrained_model: Name of the pre-trained model to use
            batch_size: Batch size for encoding
            device: Device to run the model on ("cpu" or GPU index)
            pooling: Pooling method for transformer embeddings
            return_tensors: Return type of the embeddings ("pt" for PyTorch tensors, "np" for NumPy arrays)
            get_cellosaurus_descriptions: Whether to get descriptions from Cellosaurus and encode those
            min_similarity_score: Minimum similarity score for fuzzy matching
            embeddings: Precomputed embeddings
            model: Pre-trained transformer model
            tokenizer: Tokenizer for the transformer model
            load_from_cache: Whether to load embeddings from cache
            filename: Path to the file containing embeddings
            cache_dir: Directory to store cached embeddings
        """
        # Set default filename based on embeddings_type if not provided
        if filename is None:
            filename = f"cell_embeddings_{embeddings_type}.npz"
        
        super().__init__(
            embeddings=embeddings,
            model=model,
            tokenizer=tokenizer,
            load_from_cache=load_from_cache,
            filename=filename,
            cache_dir=cache_dir,
        )

        # Store embedding configuration
        self.embeddings_type = embeddings_type
        self.pretrained_model = pretrained_model
        self.batch_size = batch_size
        self.device = device
        self.pooling = pooling
        self.return_tensors = return_tensors
        self.get_cellosaurus_descriptions = get_cellosaurus_descriptions
        self.min_similarity_score = min_similarity_score

        # Initialize Cellosaurus data (existing code)
        logging.debug("Loading Cellosaurus data...")
        cellosaurus_text = self._get_cellosaurus_text(cache_dir=cache_dir)
        logging.debug("Parsing Cellosaurus data...")
        cell_lines = self._parse_cellosaurus_text(cellosaurus_text)
        self.cell2description = {}
        self.cell2data = {}
        logging.debug(f"Processing {len(cell_lines)} cell lines from Cellosaurus...")
        for cell_line in cell_lines:
            cell_data, cell_descr = self.clean_cell_line_cellosaurus_entry(cell_line)
            self.cell2data[cell_line['ID']] = cell_data
            self.cell2description[cell_line['ID']] = cell_descr

        # Map all synonyms to the main ID
        self.synonym2cell_line = {}
        for cell_id, cell_data in self.cell2data.items():
            if 'SY' in cell_data:
                for synonym in cell_data['SY']:
                    synonym = synonym.strip()
                    if synonym and synonym not in self.synonym2cell_line:
                        self.synonym2cell_line[synonym] = cell_id

        # Initialize type-specific components
        self.onehot_encoder = None
        if embeddings_type == "one_hot":
            if onehot_enc_kwargs is None:
                raise ValueError("onehot_enc_kwargs must be provided for one_hot embeddings")
            default_onehot_enc_kwargs = {
                "handle_unknown": "ignore",
            }
            self.onehot_encoder = OneHotEncoder(**{**default_onehot_enc_kwargs, **onehot_enc_kwargs})
            X = self.get_cell_lines() + list(self.synonym2cell_line.keys())
            self.onehot_encoder.fit(np.array(X).reshape(-1, 1))

    def get_cell_lines(self) -> List[str]:
        """ Get all cell lines available in the embeddings. """
        return list(self.cell2description.keys())

    def get_cell_line_data(self, cell_line: str) -> Dict[str, Union[str, List[str]]]:
        """ Get data for a specific cell line.
        
        Args:
            cell_line (str): Cell line ID or name.
        
        Returns:
            Dict[str, Union[str, List[str]]]: Data for the cell line.
        """
        if cell_line in self.cell2data:
            return self.cell2data[cell_line]
        elif cell_line in self.synonym2cell_line:
            return self.cell2data[self.synonym2cell_line[cell_line]]
        else:
            raise ValueError(f"Cell line {cell_line} not found in the embeddings.")

    def get_cell_line_description(self, cell_line: str) -> str:
        """ Get the description for a specific cell line.
        
        Args:
            cell_line (str): Cell line ID or name.
        
        Returns:
            str: Description of the cell line.
        """
        if cell_line in self.cell2description:
            return self.cell2description[cell_line]
        elif cell_line in self.synonym2cell_line:
            return self.cell2description[self.synonym2cell_line[cell_line]]
        else:
            raise ValueError(f"Cell line {cell_line} not found in the embeddings.")

    @staticmethod
    def _get_cellosaurus_text(cache_dir: Union[str, Path] = None) -> str:
        """ Download the Cellosaurus text file and return its content. """
        if cache_dir is None:
            cache_dir = get_cache_dir()

        filepath = Path(cache_dir) / "cellosaurus.txt"
        if filepath.exists():
            with open(filepath, 'r') as file:
                return file.read()

        url = "https://ftp.expasy.org/databases/cellosaurus/cellosaurus.txt"
        response = requests.get(url)
        if response.status_code == 200:
            with open(filepath, 'w') as file:
                file.write(response.text)
                return response.text
        else:
            raise ValueError(f"Failed to download Cellosaurus text file. Status code: {response.status_code}")

    @staticmethod
    def _parse_cellosaurus_text(
            cellosaurus_text: str,
    ) -> List[Dict[str, Union[str, List[str]]]]:
        """ Parse a Cellosaurus text file and return a list of cell line entries.

        Args:
            cellosaurus_text (str): Content of the Cellosaurus text file.

        Returns:
            List[Dict[str, Union[str, List[str]]]]: List of dictionaries containing cell line information. Keys include:
                - 'ID': Cell line ID
                - 'AC': Accession number
                - 'SY': Cell line name
                - 'DR': List of database references
                - 'RX': List of references
                - 'CC': List of comments
                - 'OX': Organism
                - 'HI': Hierarchy information
                - 'CA': Cell line characteristics
                - 'DT': Date of last update
        """
        lines = cellosaurus_text.splitlines()

        cell_lines = []
        cell_line_entry = {}
        for line in lines:
            if line.startswith("ID   "):
                if cell_line_entry:
                    cell_lines.append(cell_line_entry)
                    cell_line_entry = {}
                cell_line_entry['ID'] = line[5:].strip()
            elif line.startswith("AC   "):
                cell_line_entry['AC'] = line[5:].strip()
            elif line.startswith("SY   "):
                cell_line_entry['SY'] = line[5:].strip()
            elif line.startswith("DR   "):
                cell_line_entry.setdefault('DR', []).append(line[5:].strip())
            elif line.startswith("RX   "):
                cell_line_entry.setdefault('RX', []).append(line[5:].strip())
            elif line.startswith("CC   "):
                cell_line_entry.setdefault('CC', []).append(line[5:].strip())
            elif line.startswith("OX   "):
                cell_line_entry['OX'] = line[5:].strip()
            elif line.startswith("HI   "):
                cell_line_entry['HI'] = line[5:].strip()
            elif line.startswith("CA   "):
                cell_line_entry['CA'] = line[5:].strip()
            elif line.startswith("DT   "):
                cell_line_entry['DT'] = line[5:].strip()
            # Add similar elif blocks for other line codes as needed

        # Add the last entry
        if cell_line_entry:
            cell_lines.append(cell_line_entry)

        return cell_lines

    @staticmethod
    def clean_cell_line_cellosaurus_entry(cell_line, cc_headers_to_ignore=None, unique_columns_ranking=None):
        """
        Clean and process a single cell line entry from Cellosaurus data.
        
        Args:
            cell_line (dict): Single cell line entry from parse_cellosaurus_text
            cc_headers_to_ignore (list): List of CC headers to ignore during processing
            unique_columns_ranking (list): Ordered list of columns by uniqueness ranking
        
        Returns:
            tuple: (cleaned_cell_data_dict, description_string)
        """
        
        if cc_headers_to_ignore is None:
            cc_headers_to_ignore = [
                'Miscellaneous',
                'From',
                'Anecdotal',
                'Misspelling',
                'Part of',
                'Registration',
                'Discontinued',
            ]
        
        if unique_columns_ranking is None:
            # Default ranking based on your analysis - you may want to update this
            unique_columns_ranking = [
                'Genome ancestry', 'Karyotypic information', 'Senescence', 
                'Biotechnology', 'Virology', 'Caution', 'Donor information',
                'Sequence variation', 'Characteristics', 'Transfected with',
                'Monoclonal antibody target', 'HLA typing', 'Knockout cell',
                'Microsatellite instability', 'HI', 'Breed/subspecies',
                'Derived from site', 'Population', 'Group', 
                'Monoclonal antibody isotype', 'Cell type', 'Transformant',
                'Selected for resistance to', 'CA'
            ]
        
        # Step 1: Process CC comments into separate columns
        cell_data = cell_line.copy()
        for comment in cell_data.get('CC', []):
            cc_header = comment.split(':')[0].strip()
            if cc_header not in cc_headers_to_ignore:
                cc_text = comment.split(':')[1].strip()
                cell_data[cc_header] = cell_data.get(cc_header, '') + cc_text + ' '
        
        # Step 2: Remove unwanted fields
        fields_to_ignore = ['CC', 'DT', 'SY']
        
        # Step 3: Remove features to ignore
        features_to_ignore = [
            'Problematic cell line',
            'Omics',
            'AC',
            'OX',
            'Doubling time',
        ]
        
        # Step 4: Generate description from ranked columns
        cell_description = ""
        for col in unique_columns_ranking:
            if col in fields_to_ignore or col in features_to_ignore:
                continue
            if col in cell_data and cell_data.get(col) is not None:
                cell_description += f"{cell_data[col].strip()}"
                cell_description += '\n'
        
        # Step 5: Clean description text
        # Remove PubMed references
        cell_description = re.sub(r'\(PubMed=.*?\)', '', cell_description)
        # Remove UBERON references
        cell_description = re.sub(r'UBERON=.*?\.', '', cell_description)
        # Clean up whitespace
        cell_description = cell_description.strip()
        cell_description = cell_description.replace(' .', '.')
        cell_description = cell_description.replace('  ', ' ')

        # Step 6: Clean cell synonyms
        if 'SY' in cell_data:
            cell_data['SY'] = cell_data['SY'].split(';')
            cell_data['SY'] = [syn.strip() for syn in cell_data['SY'] if syn.strip()]

        return cell_data, cell_description

    def get_fuzzy_cell_line(
            self,
            cell_line: str,
            min_similarity_score: float = 90,
            get_list: bool = False,
    ) -> Tuple[str, float]:
        """ Get the closest matching cell line ID among the available cell lines.
        
        Args:
            cell_line (str): Cell line ID or name.

        Returns:
            str: Closest matching cell line ID.
        """
        all_cell_lines = list(self.cell2description.keys())
        all_synonyms = list(self.synonym2cell_line.keys())
        if not get_list:
            closest_match, score = process.extractOne(
                cell_line,
                all_cell_lines + all_synonyms,
            )
            if score > min_similarity_score:
                if closest_match in self.cell2description:
                    # logging.debug(f"Using exact match '{closest_match}' for cell line '{cell_line}' with score {score}.")
                    return closest_match, score
                else:
                    # If the closest match is a synonym, return the corresponding cell line ID
                    closest_synonym = self.synonym2cell_line.get(closest_match, closest_match)
                    logging.debug(f"Using synonym '{closest_match}' for cell line '{cell_line}' with score {score}.")
                    return closest_synonym, score
        else:
            matches = process.extract(
                cell_line,
                all_cell_lines + all_synonyms,
                limit=None,
            )
            filtered_matches = [(m[0], m[1]) for m in matches if m[1] >= min_similarity_score]
            if filtered_matches:
                return filtered_matches, 0
            else:
                logging.debug(f"No matches found for cell line '{cell_line}' with minimum score {min_similarity_score}.")
        return cell_line, 0

    def get_cell_description(
            self,
            cell_line: str,
            use_fuzzy_matching: bool = False,
            min_similarity_score: float = 90,
            passthrough_if_not_found: bool = True,
    ) -> str:
        """ Get the description of a cell line.
        
        Args:
            cell_line (str): Cell line ID or name.
            use_fuzzy_matching (bool): Whether to use fuzzy matching if exact match not found
            min_similarity_score (float): Minimum similarity score for fuzzy matching. Must be between 0 and 100.
            passthrough_if_not_found (bool): If True, return the input cell_line if not found; otherwise raise an error.
        
        Returns:
            str: Description of the cell line.
        """
        if not (0 <= min_similarity_score <= 100):
            raise ValueError("min_similarity_score must be between 0 and 100.")
        not_found = f"Cell line {cell_line} not found in the embeddings."
        if cell_line in self.cell2description:
            return self.cell2description.get(cell_line, not_found)
        elif cell_line in self.synonym2cell_line:
            cell_id = self.synonym2cell_line[cell_line]
            return self.cell2description.get(cell_id, not_found)
        elif use_fuzzy_matching:
            closest_synonym, score = self.get_fuzzy_cell_line(
                cell_line=cell_line,
                min_similarity_score=min_similarity_score,
            )
            descr = self.cell2description.get(closest_synonym, not_found)
            # logging.debug(f"Closest match for {cell_line}: {closest_synonym} with score {score}.")
            return descr
        elif passthrough_if_not_found:
            return cell_line
        else:
            raise ValueError(f"Cell line \"{cell_line}\" not found in the embeddings.")

    def __getitem__(self, key: str) -> np.ndarray:
        """ Get the embedding for a given cell line.
        
        Args:
            key (str): Cell line ID or name.
        
        Returns:
            np.ndarray: Embedding for the cell line.
        """
        if key in self.embeddings:
            return self.embeddings[key]
        else:
            return self.embeddings[self.get_fuzzy_cell_line(key)[0]]

    def encode(
        self,
        cell_lines: Union[str, List[str]],
        skip_existing: bool = True,
        update_cache: bool = False,
    ) -> Dict[str, Union[np.array, torch.Tensor]]:
        """ Encode cell lines into embeddings using the configured method.
        
        Args:
            cell_lines: Cell line string or list of cell line strings
            skip_existing: Whether to skip already encoded cell lines
            update_cache: Whether to update the cache with new embeddings
    
        Returns:
            Dict[str, Union[np.array, torch.Tensor]]: Encoded embeddings
        """
        if isinstance(cell_lines, str):
            cells_list = [cell_lines]
        elif isinstance(cell_lines, list):
            cells_list = cell_lines
        else:
            raise ValueError("Input cell_lines must be a string or a list of strings.")
    
        # Modify the cell lines to match the existing embeddings using fuzzy matching
        if self.min_similarity_score > 0:
            cells_list = [self.get_fuzzy_cell_line(s, self.min_similarity_score)[0] for s in cells_list]
    
        # Set aside the cell lines that are already encoded
        if skip_existing:
            cells_to_encode = [s for s in cells_list if s not in self.embeddings]
            cells_encoded = {s: self.embeddings[s] for s in cells_list if s in self.embeddings}
        else:
            cells_to_encode = cells_list
    
        if not cells_to_encode:
            embeddings = {}
        else:
            embeddings = self._encode_cell_lines(cells_to_encode)
    
        if skip_existing:
            all_embeddings = {**cells_encoded, **embeddings}
            embeddings = {s: all_embeddings[s] for s in cells_list}
    
        # Update instance embeddings
        if len(embeddings) > 0:
            self.embeddings.update(embeddings)
    
        if update_cache:
            self.save()
    
        return embeddings

    def _encode_cell_lines(self, cells_to_encode: List[str]) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """ Internal method to encode cell lines based on the configured embeddings_type. """
        # Get Cellosaurus descriptions if configured to do so
        if self.get_cellosaurus_descriptions:
            original_cells = cells_to_encode
            cells_to_encode = [self.get_cell_description(s) for s in cells_to_encode]
            logging.debug(f"Encoded {len(cells_to_encode)} cell lines with descriptions.")
        else:
            original_cells = cells_to_encode
    
        if self.embeddings_type == "one_hot":
            return self._encode_one_hot(cells_to_encode, original_cells)
        elif self.embeddings_type == "transformer":
            return self._encode_with_transformer(cells_to_encode, original_cells)
        elif self.embeddings_type == "sentence_transformer":
            return self._encode_with_sentence_transformer(cells_to_encode, original_cells)
        else:
            raise ValueError(f"Unsupported embeddings_type: {self.embeddings_type}")

    def _encode_one_hot(self, cells_to_encode: List[str], original_cells: List[str]) -> Dict[str, np.ndarray]:
        """ Encode cell lines using one-hot encoding. """
        embeddings = np.array(cells_to_encode).reshape(-1, 1)
        embeddings = self.onehot_encoder.transform(embeddings).toarray()
        
        logging.debug(f"One-hot encoded embeddings shape: {embeddings.shape}")
        
        # Map to original cell names
        return {s: e for s, e in zip(original_cells, embeddings)}

    def _encode_with_transformer(self, cells_to_encode: List[str], original_cells: List[str]) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """ Encode cell lines using transformer model. """
        embeddings = self.encode_with_transformer(
            strings=cells_to_encode,
            tokenizer=self.tokenizer,
            model=self.model,
            pretrained_model=self.pretrained_model,
            batch_size=self.batch_size,
            device=self.device,
            pooling=self.pooling,
            return_tensors=self.return_tensors,
            return_dict=True,
        )
        
        # Map to original cell names
        return {s: e for s, e in zip(original_cells, embeddings.values())}

    def _encode_with_sentence_transformer(self, cells_to_encode: List[str], original_cells: List[str]) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """ Encode cell lines using sentence transformer model. """
        # Use self.model if available, otherwise load the model
        if self.model is None:
            model = SentenceTransformer(self.pretrained_model)
        else:
            model = self.model

        embeddings = model.encode(
            cells_to_encode,
            batch_size=self.batch_size,
            device=self.device,
            output_value="token_embeddings",
        )
        logging.debug(f"Embeddings shapes: {', '.join([str(e.shape) for e in embeddings])}")

        # Pool the token embeddings
        if self.pooling == "sum":
            embeddings = [e.sum(axis=0) for e in embeddings]
        elif self.pooling == "mean":
            embeddings = [e.mean(axis=0) for e in embeddings]
        elif self.pooling == "max":
            embeddings = [e.max(axis=0) for e in embeddings]
        elif self.pooling == "mean_sqrt_len":
            embeddings = [e.mean(axis=0) / np.sqrt(e.shape[0]) for e in embeddings]
        else:
            raise ValueError(f"Unsupported pooling method for sentence transformer: {self.pooling}")

        if self.return_tensors == "np":
            embeddings = [e.cpu().numpy() if isinstance(e, torch.Tensor) else e for e in embeddings]

        logging.debug(f"Embeddings shapes after pooling: {', '.join([str(e.shape) for e in embeddings])}")

        # Map to original cell names
        return {s: e for s, e in zip(original_cells, embeddings)}