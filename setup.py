from setuptools import setup, Extension

setup(
    name="neuro_ensemble",
    version="0.0.0",
    install_requires=[
        "numpy",
        "matplotlib",
        "joblib",
        "ensemble @ git+https://github.com/Anessivan/Neuro@master",
    ],
)
