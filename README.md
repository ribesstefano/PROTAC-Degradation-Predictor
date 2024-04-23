# PROTAC-Degradation-Predictor

Predicting PROTAC protein degradation activity via machine learning.

## Data Curation

For data curation code, please refer to the code in the Jupyter notebooks [`data_curation.ipynb`](notebooks/data_curation.ipynb).

## Installing the Package

To install the package, run the following command:

```bash
pip install .
```

## Running the Package

To run the package after installation, here is an example snippet:

```python
import protac_degradation_predictor as pdp

protac_smiles = 'CC(C)(C)OC(=O)N1CCN(CC1)C2=CC(=C(C=C2)C(=O)NC3=CC(=C(C=C3)F)Cl)C(=O)NC4=CC=C(C=C4)F'
e3_ligase = 'VHL'
target_uniprot = 'P04637'
cell_line = 'HeLa'

active_protac = pdp.is_protac_active(
    protac_smiles,
    e3_ligase,
    target_uniprot,
    cell_line,
    device='gpu', # Default to 'cpu'
    proba_threshold=0.5, # Default value
)

print(f'The given PROTAC is: {"active" if active_protac else "inactive"}')
```

> If you're coming from my [thesis repo](https://github.com/ribesstefano/Machine-Learning-for-Predicting-Targeted-Protein-Degradation), I just wanted to create a separate and "less generic" repo for fast prototyping new ideas.
> Stefano.
