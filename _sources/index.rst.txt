.. PROTAC-Degradation-Predictor documentation master file, created by
   sphinx-quickstart on Mon Aug 23 17:31:15 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

===========================================================
PROTAC-Degradation-Predictor: Documentation and Overview
===========================================================

**PROTAC-Degradation-Predictor** is a Python package designed to predict the activity of PROTAC molecules using advanced machine learning techniques. The tool aims to assist researchers in evaluating the potential effectiveness of PROTACs, a novel class of drugs that target protein degradation.

The package Github repository can be found `here <https://github.com/ribesstefano/PROTAC-Degradation-Predictor.git>`_.

.. .. image:: https://yourimageurl.com/logo.png  # Add your project's logo or any relevant image
..    :align: center

Introduction
============

PROTACs (Proteolysis Targeting Chimeras) are a class of molecules that induce the degradation of specific proteins. This package allows researchers to predict the activity of PROTACs by leveraging a variety of machine learning models, including XGBoost and PyTorch-based pretrained neural networks.

The primary functionalities of this package include:
- Predicting PROTAC activity using different machine learning models.
- Accessing curated datasets for training and evaluation.
- Hyperparameter tuning and model training using Optuna.

Features
========

- **Machine Learning Models**: Utilize XGBoost, PyTorch, and scikit-learn models to predict PROTAC activity (refer to the :func:`protac_degradation_predictor.protac_degradation_predictor.get_protac_active_proba` function).
- **Dataset Handling**: Load and manage datasets specific to PROTAC research (refer to the :func:`protac_degradation_predictor.data_utils.load_curated_dataset` function).
- **Customizability**: Tune model hyperparameters and experiment with different model configurations (refer to the :func:`protac_degradation_predictor.optuna_utils.hyperparameter_tuning_and_training` function).

Quickstart
==========

To get started with PROTAC-Degradation-Predictor, follow these steps:

1. **Installation**:
   Install the package using pip:
   
   .. code-block:: bash
      pip install git+https://github.com/ribesstefano/PROTAC-Degradation-Predictor.git

2. **Basic Usage**:
   Here's an example of how to predict PROTAC activity:
   
   .. code-block:: python
   
      from protac_degradation_predictor import get_protac_active_proba
      
      smiles = "CC(C)C1=CC=C(C=C1)C2=NC3=CC=CC=C3C(=O)N2"
      e3_ligase = "Q9Y6K9"
      target_uniprot = "P04637"
      cell_line = "HCT116"
      
      prediction = get_protac_active_proba(
          protac_smiles=smiles,
          e3_ligase=e3_ligase,
          target_uniprot=target_uniprot,
          cell_line=cell_line,
          device='cpu',
          use_models_from_cv=False,
          use_xgboost_models=True,
          study_type='standard'
      )
      print(prediction)

For more detailed usage and customization, refer to the relevant sections below.

Contents
========

.. toctree::
   :maxdepth: 2
   :caption: Documentation Contents:

   source/modules
   source/protac_degradation_predictor
   source/protac_degradation_predictor.optuna_utils
   source/protac_degradation_predictor.protac_dataset
   source/protac_degradation_predictor.pytorch_models

Getting Help
============

If you encounter any issues or have questions, please refer to the following resources:

- **Documentation**: Full API documentation and user guide.
- **GitHub Issues**: Report bugs or request features on the `GitHub Issues <https://github.com/ribesstefano/PROTAC-Degradation-Predictor/issues>`_ page.
- **Contributing**: Learn how to contribute to the project by reading our `Contribution Guidelines <https://github.com/ribesstefano/PROTAC-Degradation-Predictor/blob/main/CONTRIBUTING.md>`_.

License
=======

This project is licensed under the MIT License. See the `LICENSE <https://github.com/ribesstefano/PROTAC-Degradation-Predictor/blob/main/LICENSE>`_ file for details.

About
=====

**Author**: Stefano Ribes

**Version**: v1.0.2

Built with Sphinx using the `Read the Docs theme <https://sphinx-rtd-theme.readthedocs.io/>`_.

----------

*This documentation was last updated on August 27, 2024.*

