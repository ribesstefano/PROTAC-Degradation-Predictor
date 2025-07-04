from dataclasses import dataclass, field

@dataclass(frozen=True)
class Config:
    # Embeddings information
    morgan_radius: int = 10 # 15
    fingerprint_size: int = 256 # 224
    protein_embedding_size: int = 1024
    cell_embedding_size: int = 768

    # Data information
    dmax_threshold: float = 0.6
    pdc50_threshold: float = 6.0
    active_label: str = field(default=f"Active (Dmax {dmax_threshold}, pDC50 {pdc50_threshold})")
    e3_ligase2uniprot: dict = field(default_factory=lambda: {
            "VHL": "P40337",
            "CRBN": "Q96SW2",
            "DCAF11": "Q8TEB1",
            "DCAF15": "Q66K64",
            "DCAF16": "Q9NXF7",
            "MDM2": "Q00987",
            "Mdm2": "Q00987",
            "XIAP": "P98170",
            "cIAP1": "Q7Z460",
            "IAP": "P98170",  # I couldn't find the Uniprot ID for IAP, so it's XIAP instead
            "Iap": "P98170",  # I couldn't find the Uniprot ID for IAP, so it's XIAP instead
            "AhR": "P35869",
            "RNF4": "P78317",
            "RNF114": "Q9Y508",
            "FEM1B": "Q9UK73",
            "Ubr1": "Q8IWV7",
        })
    
    # URL for downloading precomputed embeddings
    drive_base_url: str = "https://docs.google.com/uc?export=download&id="
    cell2embedding_url: str = f"{drive_base_url}1N5yvFaQ7t1y3rMxinlwKFb_G-vsM13ZD"
    models_url: str = f"{drive_base_url}16uT0q30n_7cKoAmTSnJTvocWmTP7uvlL"
    curated_dataset_url: str = f"{drive_base_url}1lly2i0iUF_nqn1PmQ8SlidKZcp8dNEcF"
    uniprot2embedding_url: str = f"{drive_base_url}1U9kOVwpJFmDKc25ehAgKQDDj_AsqJxfm"

config = Config()