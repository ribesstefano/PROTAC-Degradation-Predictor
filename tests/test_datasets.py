import pandas as pd
import numpy as np

from protac_degradation_predictor.data.datasets import MolPoiE3CellDataset

def test_mol_poi_e3_cell_dataset():
    """Test the MolPoiE3CellDataset class."""
    # Create a dummy dataframe
    data = {
        "mol": [
            "CC1=C(C2=CC=C(CNC(=O)[C@@H]3C[C@@H](O)CN3C(=O)[C@@H](NC(=O)CCCNC(=O)C3=CC=C(C4=CN=C(NCC5=CC=CO5)N5C=NN=C45)C=C3)C(C)(C)C)C=C2)SC=N1",
            "CC1=C(C2=CC=C(CNC(=O)[C@@H]3C[C@@H](O)CN3C(=O)[C@H](C(C)C)N3CC4=CC=CC=C4C3=O)C(OCCNC(=O)C3=CC=CC=C3NC(=O)[C@H](CCCCN)NC(=O)[C@@H]3CCCN3C(=O)CC3=CC=CC4=CC=CC=C34)=C2)SC=N1",
            "CC1=C(C2=CC=C([C@H](C)NC(=O)[C@@H]3C[C@@H](O)CN3C(=O)[C@@H](NC(=O)CCCCCN3C=C(C4=CC(C5=CC=CC=C5O)=NN=C4N)C=N3)C(C)(C)C)C=C2)SC=N1",
        ],
        "poi": ["AAAAAAAA", "BBBBBBBBBB", "CCCCCCCCCC"],
        "e3": ["DDDDDDDDDD", "EEEEEEEEEE", "FFFFFFFFFF"],
        "cell": ["hela", "RS4; 11", "ramos"],
        "label_bin": [0, 1, 0],
        "label_reg": [0.1, None, 0.3],
        "label_multiclass": ["class1", "class2", "class1"],
    }
    df = pd.DataFrame(data)

    # Create an instance of MolPoiE3CellDataset
    ds = MolPoiE3CellDataset(
        df=df,
        mol_column="mol",
        poi_column="poi",
        e3_column="e3",
        cell_column="cell",
        label_columns=["label_bin", "label_reg", "label_multiclass"],
        save_embeddings_to_cache=False,
    )

    # Check the length of the dataset
    assert len(ds) == 3, "Dataset length should be 3"

    # Check the first item in the dataset
    item = ds[0]
    assert isinstance(item, dict), "Item should be a dictionary"
    assert "mol" in item, f"Item should contain 'mol' key: {item.keys()}"
    assert "poi" in item, f"Item should contain 'poi' key: {item.keys()}"
    assert "e3" in item, f"Item should contain 'e3' key: {item.keys()}"
    assert "cell" in item, f"Item should contain 'cell' key: {item.keys()}"
    assert "label_bin" in item, f"Item should contain 'label_bin' key: {item.keys()}"
    assert "label_reg" in item, f"Item should contain 'label_reg' key: {item.keys()}"
    assert "label_multiclass" in item, f"Item should contain 'label_multiclass' key: {item.keys()}"