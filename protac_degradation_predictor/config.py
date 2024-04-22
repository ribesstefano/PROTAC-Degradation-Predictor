from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    # Embeddings information
    morgan_radius: int = 15
    fingerprint_size: int = 224
    protein_embedding_size: int = 1024
    cell_embedding_size: int = 768

    # Data information
    dmax_threshold: float = 0.6
    pdc50_threshold: float = 6.0
    e3_ligase2uniprot: dict = {
        'VHL': 'P40337',
        'CRBN': 'Q96SW2',
        'DCAF11': 'Q8TEB1',
        'DCAF15': 'Q66K64',
        'DCAF16': 'Q9NXF7',
        'MDM2': 'Q00987',
        'Mdm2': 'Q00987',
        'XIAP': 'P98170',
        'cIAP1': 'Q7Z460',
        'IAP': 'P98170',  # I couldn't find the Uniprot ID for IAP, so it's XIAP instead
        'Iap': 'P98170',  # I couldn't find the Uniprot ID for IAP, so it's XIAP instead
        'AhR': 'P35869',
        'RNF4': 'P78317',
        'RNF114': 'Q9Y508',
        'FEM1B': 'Q9UK73',
        'Ubr1': 'Q8IWV7',
    }

    def __post_init__(self):
        self.active_label: str = f'Active (Dmax {self.dmax_threshold}, pDC50 {self.pdc50_threshold})'


config = Config()