from setuptools import setup, find_packages

setup(
    name="neuro_ensemble",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "joblib",
        "ensemble @ git+https://github.com/Anessivan/Neuro@master",
    ],
)
