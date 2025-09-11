""" Module for handling protein information and sequences. """
import re
from typing import Optional, Union, Literal
import time
from functools import lru_cache
import requests


@lru_cache()
def fetch_uniprot_entry(uniprot_id: str) -> Optional[dict]:
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        time.sleep(0.5)  # To avoid hitting the API too hard
        return r.json()
    except Exception as e:
        print(f"[UniProt] fetch failed for {uniprot_id}: {e}")
        return None

def extract_protein_info(uniprot_id: str, skip_isoforms: bool = False) -> Optional[dict]:
    """ Extracts detailed information about a protein from UniProt.
    
    Args:
        uniprot_id (str): The UniProt ID of the protein.
        skip_isoforms (bool): If True, skips fetching isoform information.
        
    Returns:
        dict: A dictionary containing the protein information, or None if the fetch fails. List of keys:
            - 'accession': Primary accession number.
            - 'secondary_accessions': List of secondary accession numbers.
            - 'sequence': Canonical sequence of the protein.
            - 'full_names': List of full names of the protein.
            - 'short_names': List of short names of the protein.
            - 'isoforms': List of isoform information (if not skipped).
            - 'locations': List of subcellular locations.
            - 'natural_variants': List of natural variant sequences.
            - 'natural_variants_ids': List of IDs for the natural variants.
    """
    # Fetch the UniProt entry
    entry = fetch_uniprot_entry(uniprot_id)
    if not entry:
        print(f"[UniProt] {uniprot_id} fetch failed.")
        return None

    # Setup the information dictionary to return
    info = {
        'accession': entry.get('primaryAccession'),
        'secondary_accessions': entry.get('secondaryAccessions'),
        'sequence': entry.get('sequence', {}).get('value'),
    }

    # Obtain full names and short names
    alternative_names = entry.get('proteinDescription', {}).get('alternativeNames', [])
    info['full_names'] = [n.get('fullName', {}).get('value', 'N/A') for n in alternative_names]
    info['short_names'] = [n.get('value', 'N/A') for an in alternative_names for n in an.get('shortNames', [])]

    # Parse comments for isoforms and locations in cell
    info['isoforms'] = []
    info['locations'] = []
    comments = entry.get('comments', [])
    for comment in comments:
        # Get isoforms IDs if present, they will be recursively fetched later
        if comment.get('commentType', '') == 'ALTERNATIVE PRODUCTS':
            if not skip_isoforms:
                for isoform in comment.get('isoforms', []):
                    if isoform.get('isoformIds'):
                        info['isoforms'] += isoform['isoformIds']

        # Get subcellular locations
        elif comment.get('commentType', '') == 'SUBCELLULAR LOCATION':
            for location in comment.get('subcellularLocations', []):
                location = location.get('location', {})
                if location.get('value'):
                    info['locations'].append(location['value'])
    
    # Ensure locations are lowercase and unique
    locations = []
    for loc in info['locations']:
        for l in loc.split(', '):
            l = l.strip().lower()
            if l not in locations:
                locations.append(l)
    info['locations'] = locations

    info['natural_variants'] = []
    info['natural_variants_ids'] = []
    features = entry.get('features', [])
    for feature in features:
        if feature.get('type', '') == 'Natural variant':
            loc = feature.get('location', {})
            start = loc.get('start', {}).get('value')
            end = loc.get('end', {}).get('value')
            if start is not None and end is not None:
                alt_seq_info = feature.get('alternativeSequence', {})
                original_seq = alt_seq_info.get('originalSequence', '')
                alt_seq = ''.join(alt_seq_info.get('alternativeSequences', ['']))
                if info['sequence'][start-1:end] != original_seq:
                    print(f"[WARNING] Sequence mismatch for {uniprot_id} at {start}-{end}: {info['sequence'][start-1:end]} != {original_seq}")
                natural_variant = (
                    info['sequence'][:start-1] + alt_seq + info['sequence'][end:]
                )
                info['natural_variants'].append(natural_variant)
                info['natural_variants_ids'].append(feature.get('featureId'))

    # If isoforms are not skipped, fetch their details recursively
    # NOTE: Recursion is disabled within an isoform extraction.
    info['isoforms'] = [extract_protein_info(iso_id, skip_isoforms=True) for iso_id in info['isoforms']]
    info['isoforms'] = [iso for iso in info['isoforms'] if iso is not None]
    
    return info

def apply_mutation(
        seq: str,
        gene: str,
        uniprot: Optional[str] = None,
        on_error: Union[bool, Literal['raise', 'ignore']] = 'raise',
        verbose: int = 0,
) -> str:
    """ Apply the mutation to the sequence, if possible.
    
    Args:
        uniprot (str): The UniProt ID of the protein.
        gene (str): The gene name or mutation description.
        seq (str): The original protein sequence.
        on_error (str): What to do on error ('raise' or 'ignore').
        
    Returns:
        str: The mutated sequence if the mutation is valid, otherwise the original sequence.
        
    Raises:
        ValueError: If the mutation cannot be applied and `on_error` is 'raise'.
    """
    # # TODO: Just use a dictionary and replace these sequences straightaway...
    # uniprot_exceptions = {
    #     ('O60885', 'BRD4 BD1'): uniprot2sequence['O60885'],
    #     ('P25440', 'BRD2 BD2'): uniprot2sequence['P25440'],
    #     ('P10275', 'AR-V7'): uniprot2sequence['P10275'],
    #     # TODO: Not working... why???
    #     ('P00533', 'EGFR e19d'): uniprot2sequence['P10275'],
    # }
    # # Handle exceptions
    # if (uniprot, gene) in uniprot_exceptions:
    #     return uniprot_exceptions[(uniprot, gene)]

    # Use regex to get all mutations in the gene string
    if re.search(r'\b[A-Z]\d+[A-Z]\b', gene) or re.search(r'\bDEL', gene):
        mutations = re.findall(r'\b[A-Z]\d+[A-Z]\b|\bDEL\d+\b', gene.upper())
    else:
        return seq

    if verbose > 0:
        print(f'Applying mutations: {mutations} to sequence: {seq} (length: {len(seq)})')

    original_seq = seq
    del_ops = 0
    for op in mutations:
        if 'del' in op.lower():
            idx = int(op.lower().split('del')[1]) - 1
            seq = seq[:idx] + seq[idx + 1:]
            del_ops += 1
        else:
            # Replace aminoacid at a specific index
            # NOTE: The indexing starts from one, not zero.
            curr, idx, mutation = op[0].upper(), int(op[1:-1])-1, op[-1].upper()
            # NOTE: If a deletion has happened before, the index is still
            # relative to the whole sequence lenght (weird...)
            idx -= del_ops
            if verbose > 1:
                print(f'Operation: {op} on ...{seq[idx-8:idx]}[{seq[idx]} -> {mutation}]{seq[idx+1:idx+8]}...')
            if curr != seq[idx]:
                msg = f'Replacement at position {idx} failed. Expected "{curr}", found: "{seq[idx]}".'
                if on_error == 'raise' or on_error is True:
                    raise ValueError('ERROR. ' + msg)
                else:
                    if verbose > 0:
                        print('WARNING. ' + msg + ' No mutation is applied.')
                    return original_seq
            seq = seq[:idx] + mutation + seq[idx + 1:]

    return seq