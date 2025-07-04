import setuptools

def read_requirements():
    # Read requirements from file
    with open("requirements.txt") as f:
        return f.read().splitlines()

setuptools.setup(
    name="protac_degradation_predictor",
    version="2.0.0",
    author="Stefano Ribes",
    url="https://github.com/ribesstefano/PROTAC-Degradation-Predictor",
    author_email="ribes.stefano@gmail.com",
    description="A package to predict PROTAC-induced protein degradation.",
    long_description=open("README.md").read(),
    packages=setuptools.find_packages(),
    install_requires=read_requirements(),
    classifiers=[
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    # include_package_data=True,
    # package_data={"": ["data/*.h5", "data/*.pkl", "data/*.csv", "models/*.ckpt"]},
)
