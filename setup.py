import os
from setuptools import setup, find_packages

setup(
    name='deepsport',
    author='Gabriel Van Zandycke',
    author_email="gabriel.vanzandycke@hotmail.com",
    url="https://github.com/gabriel-vanzandycke/deepsport",
    licence="LGPL",
    python_requires='>=3.8',
    description="Software made public for my PhD addressing ball detection and ball 3D localization",
    version='1.5.0',
    packages=find_packages(),
    install_requires=[
        "mlworkflow>=0.6.0",
        "dill",
        "numpy",
        "calib3d>=2.10.0",
        "pandas",
        "deepsport-utilities>=4.10.0",
        "tensorflow",
        "python-dotenv",
        "scikit-learn",
        "jupyter",
        "experimentator",# @ git+https://github.com/gabriel-vanzandycke/experimentator@main",
        "tf_layers @ git+https://github.com/gabriel-vanzandycke/tf_layers@main",
        "icnet_tf2 @ git+https://github.com/gabriel-vanzandycke/icnet_tf2@master",
    ],
)
