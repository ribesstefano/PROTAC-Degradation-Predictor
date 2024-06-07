<!-- ![Maturity level-0](https://img.shields.io/badge/Maturity%20Level-ML--0-red)

# PROTAC-Degradation-Predictor -->

<p align="center">
  <img src="https://img.shields.io/badge/Maturity%20Level-ML--0-red" alt="Maturity level-0">
</p>

<h1 align="center">PROTAC-Degradation-Predictor</h1>

<p align="center">
  A machine learning-based tool for predicting PROTAC protein degradation activity.
</p>

## 📚 Table of Contents

- [Data Curation](#-data-curation)
- [Installation](#-installation)
- [Usage](#-usage)

## 📝 Data Curation

The code for data curation can be found in the Jupyter notebook [`data_curation.ipynb`](notebooks/data_curation.ipynb).

## 🚀 Installation

To install the package, open your terminal and run the following commands:

```bash
git clone https://github.com/ribesstefano/PROTAC-Degradation-Predictor.git
cd PROTAC-Degradation-Predictor
pip install .
```

The package has been developed on a Linux machine with Python 3.10.8. It is recommended to use a virtual environment to avoid conflicts with other packages.

## 🎯 Usage

After installing the package, you can use it as follows:

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
    device='cuda', # Default to 'cpu'
    proba_threshold=0.5, # Default value
)

print(f'The given PROTAC is: {"active" if active_protac else "inactive"}')
```

This example demonstrates how to predict the activity of a PROTAC molecule. The `is_protac_active` function takes the SMILES string of the PROTAC, the E3 ligase, the UniProt ID of the target protein, and the cell line as inputs. It returns whether the PROTAC is active or not.

The function supports batch computation by passing lists of SMILES strings, E3 ligases, UniProt IDs, and cell lines. In this case, it returns a list of booleans indicating the activity of each PROTAC.




## 📈 Training

The code for training the model can be found in the file [`run_experiments.py`](src/run_experiments.py).

## 📄 Citation

If you use this tool in your research, please cite the following paper:

```
@misc{ribes2024modeling,
    title={Modeling PROTAC Degradation Activity with Machine Learning},
    author={Stefano Ribes and Eva Nittinger and Christian Tyrchan and Rocío Mercado},
    year={2024},
    eprint={2406.02637},
    archivePrefix={arXiv},
    primaryClass={q-bio.QM}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.