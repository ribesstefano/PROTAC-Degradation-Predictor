import time
import logging

from rdkit import Chem
from sklearn.preprocessing import OneHotEncoder
import numpy as np

from protac_degradation_predictor.data.embeddings.mol_embeddings import MolEmbedding
from protac_degradation_predictor.data.embeddings.cell_embeddings import CellEmbedding
from protac_degradation_predictor.data.embeddings.protein_embeddings import ProteinEmbedding


def test_mol_embedding():
    smiles = [
        'CC1=C(C2=CC=C(CNC(=O)[C@@H]3C[C@@H](O)CN3C(=O)[C@@H](NC(=O)CCCNC(=O)C3=CC=C(C4=CN=C(NCC5=CC=CO5)N5C=NN=C45)C=C3)C(C)(C)C)C=C2)SC=N1',
        'CC1=C(C2=CC=C(CNC(=O)[C@@H]3C[C@@H](O)CN3C(=O)[C@H](C(C)C)N3CC4=CC=CC=C4C3=O)C(OCCNC(=O)C3=CC=CC=C3NC(=O)[C@H](CCCCN)NC(=O)[C@@H]3CCCN3C(=O)CC3=CC=CC4=CC=CC=C34)=C2)SC=N1',
        'CC1=C(C2=CC=C([C@H](C)NC(=O)[C@@H]3C[C@@H](O)CN3C(=O)[C@@H](NC(=O)CCCCCN3C=C(C4=CC(C5=CC=CC=C5O)=NN=C4N)C=N3)C(C)(C)C)C=C2)SC=N1',
        'O=C1CCC(N2C(=O)C3=CC=CC(NC(=O)CCCCCCOC4=C(O)C=C(O)C5=C4OC(C4=CC=CC=C4)=CC5=O)=C3C2=O)C(=O)N1',
        'COC1=CC=C(C2=N[C@@H](C3=CC=C(Cl)C=C3)[C@@H](C3=CC=C(Cl)C=C3)N2C(=O)N2CCN(CC(=O)NCCOCCOCCOCCC(=O)N3CCC[C@@H](C4=CC=C(N5C=C6C=CC=C(C(N)=O)C6=N5)C=C4)C3)C(=O)C2)C(OC(C)C)=C1',
        'CC(=O)C1=C(C)C2=CN=C(NC3=CC=C(N4CCN(CC(=O)NCCCCNC5=CC=CC6=C5C(=O)N(C5CCC(=O)NC5=O)C6=O)CC4)C=N3)N=C2N(C2CCCC2)C1=O',
        'NC(=O)CC[C@H](NC(=O)[C@@H]1CC[C@@H]2CCN(C(=O)CCCCCC#CC3=CC=CC4=C3CN(C3CCC(=O)NC3=O)C4=O)C[C@H](NC(=O)C3=CC4=CC(C(F)(F)P(=O)(O)O)=CC=C4[NH]3)C(=O)N12)C(=O)NCC1=CC=CC=C1',
        'CC(C)C[C@H](NC(=O)[C@@H](O)[C@H](N)CC1=CC=CC=C1)C(=O)NCCOCCOCCOCCNC(=O)C1=CC=C(S(=O)(=O)CC(C)(O)C(=O)NC2=CC=C(C#N)C(C(F)(F)F)=C2)C=C1',
        'COC1=CC2=C(NC3=CC=CC=C3)N=CN=C2C=C1OCC1CCN(C(=O)CCCCCCCCCCCC(=O)NCCCCCC(=O)N[C@H](C(=O)N2C[C@H](O)C[C@H]2C(=O)NCC2=CC=C(C3=C(C)N=CS3)C=C2)C(C)(C)C)CC1',
        'CC1=C(C2=CC=C([C@H](C)NC(=O)[C@@H]3C[C@@H](O)CN3C(=O)[C@@H](NC(=O)CCCCNC(=O)C3=CC=CC=C3CN3CCN(S(=O)(=O)C4=CC=C(NC(=S)NCC5=CC=CN=C5)C=C4)CC3)C(C)(C)C)C=C2)SC=N1',
        'CCOC(=O)C1=C[C@@H](OC(CC)CC)[C@H](NC(C)=O)[C@@H](NCCCCC(=O)NC2=CC=CC3=C2C(=O)N(C2CCC(=O)NC2=O)C3=O)C1',
        'CC1=C(C2=CC=C(CNC(=O)[C@@H]3C[C@@H](O)CN3C(=O)[C@@H](NC(=O)CCCCCN3C=C(COC(=O)NCC4=CC=C(C(=O)NC5=CC=CC=C5N)C=C4)N=N3)C(C)(C)C)C=C2)SC=N1',
        'CN[C@@H](C)C(=O)N[C@H](C(=O)N1CCC[C@H]1C1=NC(C(=O)C2=CC=CC(OCCOCCOCCOCCOC3=CC=CC(CN4C(C)=C(C#N)C(C5=CC=C(C#N)C=C5)=C4C)=C3)=C2)=CS1)C1CCCCC1',
        'CCCS(=O)(=O)NC1=CC=C(F)C(C(=O)C2=C[NH]C3=NC=C(C4=CC=C(N5CCN(CC(=O)N[C@H](C(=O)N6C[C@H](O)C[C@H]6C(=O)NCC6=CC=C(C7=C(C)N=CS7)C=C6)C(C)(C)C)CC5)C=C4)C=C23)=C1F',
        'CC1=C(C)C2=C(S1)N1C(C)=NN=C1[C@H](CC(=O)NCCC(=O)NCC(=O)NC1CSCC3=CC=CC(=C3)CSCC(C(N)=O)NC(=O)C(CC(C)C)NC(=O)C(CC(=O)O)NC(=O)C3CCCN3C1=O)N=C2C1=CC=C(Cl)C=C1',
        'C=CC(=O)NC1=CC=CC(NC2=NC(NC3=CC=C(N4CCN(C(=O)CCCCCCCCCCC(=O)NC5=CC=CC6=C5C(=O)N(C5CCC(=O)NC5=O)C6=O)CC4)C=C3OC)=NC=C2C(F)(F)F)=C1',
        'C#CC1=C(F)C=CC2=CC(O)=CC(C3=NC=C4C(N5C[C@@H]6CC[C@H](C5)N6)=NC(OC[C@@H]5CCCN5CCCCOCC(=O)N[C@H](C(=O)N5C[C@H](O)C[C@H]5C(=O)N[C@@H](C)C5=CC=C(C6=C(C)N=CS6)C=C5)C(C)(C)C)=NC4=C3F)=C12',
        'CC1=C(C2=CC=C(CNC(=O)[C@@H]3C[C@@H](O)CN3C(=O)[C@@H](NC(=O)[C@H]3C[C@@H](NC(=O)C4=CC=C(C5=CN=C(NCC6=CC=CO6)N6C=NN=C56)C=C4)C3)C(C)(C)C)C=C2)SC=N1',
        'CN(CCOCCCC1=CC=CC2=C1C(=O)N(C1CCC(=O)NC1=O)C2=O)CC1=CC=C(N2C=C(NC(=O)C3=COC(C4=CC=NC(NCC5CC5)=C4)=N3)C(C(N)=O)=N2)C=C1',
        'CN[C@@H](C)C(=O)N[C@H](C(=O)N1C[C@@H](OC2=CC=CC(OCCCCCCOCCCCCCOCCOC3=CC(C4=C(C)N=CS4)=CC=C3CNC(=O)[C@@H]3C[C@@H](O)CN3C(=O)[C@H](C(C)C)N3CC4=CC=CC=C4C3=O)=C2)C[C@H]1C(=O)N[C@@H]1CCCC2=CC=CC=C21)C1CCCCC1',
    ]

    mol_embedding = MolEmbedding(radius=2, fp_size=1024)
    assert mol_embedding.radius == 2
    assert mol_embedding.fp_size == 1024
    assert isinstance(mol_embedding.morgan_fpgen, Chem.rdFingerprintGenerator.FingerprintGenerator64)
    
    fp_embeddings = mol_embedding.encode(smiles, embeddings_type="fingerprint", skip_existing=False, update_cache=False)
    transf_embeddings = mol_embedding.encode(smiles, embeddings_type="transformer", skip_existing=False, update_cache=False)

    assert len(fp_embeddings) == len(transf_embeddings) == len(smiles)
    assert all(isinstance(emb, np.ndarray) for emb in fp_embeddings.values())
    assert all(isinstance(emb, np.ndarray) for emb in transf_embeddings.values())

    # Timestamp the embeddings generation with Transformer model with and
    # without cache
    mol_embedding = MolEmbedding(load_from_cache=False)
    print("Generating Transformer embeddings without cache...")
    start_time = time.time()
    transf_embeddings_no_cache = mol_embedding.encode(smiles* 5, embeddings_type="transformer")
    end_time = time.time()
    print(f"Time taken to generate Transformer embeddings without cache: {end_time - start_time:.2f} seconds")

    mol_embedding = MolEmbedding()
    print("Molecular embeddings with cache loaded. Starting generating embeddings...")
    start_time = time.time()
    transf_embeddings = mol_embedding.encode(smiles* 5, embeddings_type="transformer")
    end_time = time.time()
    print(f"Time taken to generate Transformer embeddings with cache: {end_time - start_time:.2f} seconds")

    print("Molecular embeddings with cache loaded. Starting generating embeddings...")
    start_time = time.time()
    transf_embeddings = mol_embedding.encode(smiles* 5, embeddings_type="transformer")
    end_time = time.time()
    print(f"Time taken to generate Transformer embeddings with cache: {end_time - start_time:.2f} seconds")
    
    for k, v in transf_embeddings.items():
        assert isinstance(v, np.ndarray), f"Expected numpy array for {k}, got {type(v)}"
        assert k in transf_embeddings_no_cache, f"Key {k} not found in embeddings without cache"
        assert np.allclose(v, transf_embeddings_no_cache[k]), f"Embeddings for {k} do not match between cached and non-cached versions"

    for k, v in transf_embeddings_no_cache.items():
        assert isinstance(v, np.ndarray), f"Expected numpy array for {k}, got {type(v)}"
        assert k in transf_embeddings, f"Key {k} not found in embeddings with cache"
        assert np.allclose(v, transf_embeddings[k]), f"Embeddings for {k} do not match between cached and non-cached versions"


def test_cell_embedding():
    cell_lines = [
        'HeLa',
        'RI-1',
        'HEK293',
        'EOL-1',
        'A-204',
        'MCF-7',
        'T47D',
        'LNCaP',
        'VCaP AR+',
        'VCaP',
        '22Rv1',
        'NCI-H661',
        'NCI-H838',
        'A375',
        'MV4;11',
        'HEK293T',
        'MM1S',
        'MDA-MB-231',
        'HCT116',
        'Ramos',
        'THP-1',
        'SU-DHL-1',
        'NCI-H2228',
        '22RV1',
        'Jurkat',
        'U251',
        'RAMOS',
        'HBL-1',
        'Mino',
        'IgE MM',
        'PC3',
        'OVCAR8',
        'HCC827',
        'H3255',
        'H1975',
        'NAMALWA',
        'XLA',
        'THP1',
        'LnCaP95',
        'Namalwa',
        'CA-46',
        'MM.1 S',
        'MV4-11',
        'HD-MB03',
        'SUM149',
        'A549',
        'K562',
        'Molm-16',
        'SR',
        '293T',
        'H3122',
        'Karpas 299',
        'Kelly',
        'Ba/F3',
        'RS4; 11',
        'A431',
        'HCC-827',
        'PC-3',
        'MOLT-4',
        'platelets',
        'Panc Tu-I',
        'MM.1S',
        'MV-4-11',
        'NCI-H1568',
        'PBMC',
        'DB',
        'MCF7',
        'Hs578t',
        'SRD15',
        'HT-29',
        'SK-MEL-28',
        'COLO 205',
        'Mouse 4935',
        'PC9',
        'Primary Cardiomyocytes',
        'A152T neurons',
        'Huh7.5',
        'WI38',
        'KYSE520',
        'primary Sertoli',
        'germ',
        'TM3',
        'PA1',
        'MDA-MB-436',
        'SW480',
        'MB-MDA-231',
        'HLE',
        'HuH-7',
        'SNU-423',
        'HUH-1',
        'HepG2',
        'SK-Hep-1',
        'Hep3B2 1-7',
        'SNU-387',
        'HLF',
        'SNU-398',
        'HUCCT1',
        'NCI-H2030',
        'NCI-H23',
        'NCI-H358',
        'MIA PaCa-2',
        'SW1573',
        'L-O2',
        'HCC1937',
        'MDA-MB-468',
        'Capan-1',
        'SW620',
        'KM12',
        'HEL',
        'PC3-S1',
        'Sk-Mel-28',
        'RAW 264.7',
        'RS4;11',
        'NGP',
        'Pfeiffer',
        'DOHH2',
        'U937',
        'Z138',
        'OCI-AML2',
        'OCI-AML3',
        'Kasumi-1',
        'NB4',
        'H1666',
        'CAL-12T',
        'SK-MEL-246',
        'MOLM-14',
        'KYSE-270',
        'MOLM-13',
        'HCT-116',
        'COLO-205',
        'Calu-1',
        'human dermal papilla',
        'KU812',
        'BT474',
        'MIA PaCa2',
        'AsPC-1',
        'SK-LU-1',
        'HEI-OC1',
        'Bel-7402',
        'SH-SY5Y',
        'H358',
        'H23',
        'JeKo-1',
        'TMD8',
        'MHH-CALL-4',
        'PDX SJBALL020589',
        'KOPN49',
        'MOLM-16',
        'SUP-M2',
        'CCLP1',
        'KATO III',
        'BaF3 FLT3-ITD',
        'MDA-Pca-2b',
        'HEK293A',
        'HT1080',
        'T-cell',
        'HEK293-hTau',
        '231MFP',
        'HAP1',
        'ER-positive breast cancer cell lines',
        'MPro-eGFP stable',
        'CHL-1',
        'HT29',
        'SK-Mel-28',
        'SK-Mel-5',
        'H838',
        'HL-60',
        'EBC-1',
        'Hs746T',
        'AML12',
        '4T1',
        'A2780',
        'H293T',
        'CAL33',
        'HL60',
        'MOLM13',
        'MDA-MB-453',
        'OVCAR-5',
        'CAOV4',
        'Caki-1',
        'LCC2',
        'Karpas299',
        'H1650',
        'H1650R',
        'Molt-4',
        'MOLT4',
        'HAP 1',
        'CWR22Rv1',
        'KOPT-K1',
        'MV4; 11',
        'SKNO1',
        'KU812 CML',
        'MKN-45',
        'MEK1',
        'PANC1',
        'NIH-3T3',
        'U87',
        'GBM43',
        'TMD-8',
        'HGC-27',
        'HUVEC',
        'RKO',
        'Jurkat E6-1',
        'HT-1080 fibrosarcoma',
        '293FT',
        'NCI-H1703',
        'NCI-H1975',
        'Jeko-1',
        'Calu-6',
        'CAL-148',
        'KCL22',
        'Taxol',
        'Raji',
    ]
    
    cell_embedding = CellEmbedding(onehot_enc_kwargs={
        "dtype": np.float32,
        "handle_unknown": "ignore",
        "max_categories": 512,
    })
    assert isinstance(cell_embedding.onehot_encoder, OneHotEncoder)

    embeddings = cell_embedding.encode(cell_lines, embeddings_type="one_hot", skip_existing=False, update_cache=False)

    assert len(embeddings) == len(cell_lines)
    # Check that there are no None values in the embeddings
    assert all(emb is not None for emb in embeddings.values())

    print(f"Shape of embeddings: {cell_embedding.shape()}")
    # assert isinstance(cell_embedding.shape(), int) or isinstance(cell_embedding.shape(), tuple), "Shape should be an int or a tuple"


    cell_embedding = CellEmbedding(load_from_cache=False)

    cell_description = cell_embedding.get_cell_description('HeLa')
    print(f"Cell description for 'HeLa': {cell_description}")

    cell_description_hela = cell_embedding.get_cell_description('hela')
    assert cell_description == cell_description_hela, "Fuzzy matching failed for 'HeLa'"

    embeddings = cell_embedding.encode(
        cell_description,
        embeddings_type="sentence_transformer",
        skip_existing=False,
        update_cache=False,
    )

    assert len(embeddings) == 1
    assert isinstance(embeddings[cell_description], np.ndarray), "Expected numpy array for cell description embedding"

    print(f"Shape of cell description embedding: {cell_embedding.shape()}")
    print(f"Shape of cell description embedding: {cell_embedding[cell_description].shape}")

    assert isinstance(cell_embedding.embeddings, dict), "Embeddings should be a dictionary"
    assert all(isinstance(v, np.ndarray) for v in cell_embedding.embeddings.values()), "All embeddings should be numpy arrays"
    print(f"Shape of embeddings: {cell_embedding.shape()}")
    assert isinstance(cell_embedding.shape(), int) or isinstance(cell_embedding.shape(), tuple), "Shape should be an int or a tuple"
