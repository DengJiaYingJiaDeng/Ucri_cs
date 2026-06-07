from setuptools import find_packages, setup

setup(
    name="ucri-cs",
    version="0.1.0",
    packages=find_packages(where="."),
    package_dir={"": "."},
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "lightgbm>=4.0.0",
        "catboost>=1.2.0",
    ],
)
