---
title: PROTAC-Degradation-Predictor
emoji: 🧬
colorFrom: pink
colorTo: green
sdk: gradio
sdk_version: 4.37.2
app_file: app.py
pinned: false
license: mit
---

![Maturity level-0](https://img.shields.io/badge/Maturity%20Level-ML--0-red)
<!-- <a href="https://colab.research.google.com/github/ribesstefano/PROTAC-Degradation-Predictor/blob/main/notebooks/protac_degradation_predictor_tutorial.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> -->
[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/ailab-bio/PROTAC-Degradation-Predictor)

# PROTAC-Degradation-Predictor

A machine learning-based tool for predicting PROTAC protein degradation activity.

## 📚 Table of Contents

- [Data Curation](#-data-curation)
- [Installation](#-installation)
- [Documentation and Usage](#-documentation-and-usage)
- [Training](#-training)
- [Citation](#-citation)
- [License](#-license)

## 📝 Data Curation

The code for data curation can be found in the Jupyter notebook [`data_curation.ipynb`](notebooks/data_curation.ipynb).

The folder [data/studies](data/studies/) contains the training and test data used in each study reported in our paper. The label column that is used for predictions is named _"Active (Dmax 0.6, pDC50 6.0)"_ and contains binary values.

## 🚀 Installation

To install the package, open your terminal and run the following commands:

```bash
git clone --branch=main --depth=1 https://github.com/ribesstefano/PROTAC-Degradation-Predictor.git
cd PROTAC-Degradation-Predictor
pip install .
```
<!-- pip install git+https://github.com/ribesstefano/PROTAC-Degradation-Predictor.git -->

The package has been developed on a Linux machine with Python 3.10.8. It is recommended to use a virtual environment to avoid conflicts with other packages.

## 🎯 Documentation and Usage

The package documentation can be found [here](https://ribesstefano.github.io/PROTAC-Degradation-Predictor/).
For a walkthrough on how to use the package, please refer to the tutorial notebook [`protac_degradation_predictor_tutorial.ipynb`](notebooks/protac_degradation_predictor_tutorial.ipynb).

After installing the package, you can use it as follows:

```python
import protac_degradation_predictor as pdp

protac_smiles = 'Cc1ncsc1-c1ccc(CNC(=O)[C@@H]2C[C@@H](O)CN2C(=O)[C@@H](NC(=O)COCCCCCCCCCOCC(=O)Nc2ccc(C(=O)Nc3ccc(F)cc3N)cc2)C(C)(C)C)cc1'
e3_ligase = 'VHL'
target_uniprot = 'P04637'
cell_line = 'HeLa'

active_protac = pdp.is_protac_active(
    protac_smiles,
    e3_ligase,
    target_uniprot,
    cell_line,
)

print(f'The given PROTAC is: {"active" if active_protac else "inactive"}')
```

This example demonstrates how to predict the activity of a PROTAC molecule. The `is_protac_active` function takes the SMILES string of the PROTAC, the E3 ligase, the UniProt ID of the target protein, and the cell line as inputs. It returns whether the PROTAC is active or not.

The function supports batch computation by passing lists of SMILES strings, E3 ligases, UniProt IDs, and cell lines. In this case, it returns a list of booleans indicating the activity of each PROTAC.

## 📈 Training

Before running the experiments reported in our work or train on your custom dataset, here are some required steps to follow (assuming one is in the repository directory already):
1. Download the data from the [Cellosaurus database](https://web.expasy.org/cellosaurus/) and save it in the `data` directory:
```bash
wget https://ftp.expasy.org/databases/cellosaurus/cellosaurus.txt data/
```
2. Make a copy of the Uniprot embeddings to be placed in the `data` directory:
```bash
cp protac_degradation_predictor/data/uniprot2embedding.h5 data/
```
3. Create a virtual environment and install the required packages by running the following commands:
```bash
conda env create -f environment.yaml
conda activate protac-degradation-predictor
```
4. The code for training the PyTorch models can be found in the file [`run_experiments_pytorch.py`](src/run_experiments_pytorch.py).

(Don't forget to adjust the `PYTHONPATH` environment variable to include the repository directory: `export PYTHONPATH=$PYTHONPATH:/path/to/PROTAC-Degradation-Predictor`)

### Training on Custom Dataset

For training a model on a user-provided dataset, please refer to the guide reported in [this README](src/README.md).

## 📄 Citation

If you use this tool in your research, please cite the following paper:

```
@article{Ribes_2024,
   title={Modeling PROTAC degradation activity with machine learning},
   volume={6},
   ISSN={2667-3185},
   url={http://dx.doi.org/10.1016/j.ailsci.2024.100104},
   DOI={10.1016/j.ailsci.2024.100104},
   journal={Artificial Intelligence in the Life Sciences},
   publisher={Elsevier BV},
   author={Ribes, Stefano and Nittinger, Eva and Tyrchan, Christian and Mercado, Rocío},
   year={2024},
   month=dec, pages={100104}
}
```

The directories [logs](logs/) and [reports](reports/) contain the logs and reports generated during the experiments reported in the paper. Additionally, in [reports](reports/), one can find the pickled Optuna studies for the reported experiments.

The directory [models](models/) contains the trained models for the experiments reported in the paper.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
