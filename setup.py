import setuptools

setuptools.setup(
    name="protac_degradation_predictor",
    version="1.0.1",
    author="Stefano Ribes",
    url="https://github.com/ribesstefano/PROTAC-Degradation-Predictor",
    author_email="ribes.stefano@gmail.com",
    description="A package to predict PROTAC-induced protein degradation.",
    long_description=open("README.md").read(),
    packages=setuptools.find_packages(),
    install_requires=["torch", "pytorch_lightning", "scikit-learn", "imbalanced-learn", "rdkit", "pandas", "joblib", "h5py", "optuna", "torchmetrics", "xgboost"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    package_data={"": ["data/*.h5", "data/*.pkl", "data/*.csv", "models/*.ckpt"]},
)
