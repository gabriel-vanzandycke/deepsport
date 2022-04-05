This repository holds the task specific software made public for my PhD under the DeepSport project funded by the Walloon Region of Belgium, Keemotion and SportRadar.
It goes along with several open-source libraries that I developed during that time:
 - [calib3d](https://github.com/gabriel-vanzandycke/calib3d): a library to ease computations in 2D and 3D homogenous coordinates and projective geometry with camera calibration.
 - [aleatorpy](https://github.com/gabriel-vanzandycke/pseudo_random): a library to control randomness in ML experiments.
 - [experimentator](https://github.com/gabriel-vanzandycke/experimentator): a library to run DL experiments.
 - [pyconfyg](https://github.com/gabriel-vanzandycke/pyconfyg): a library to describe configuration file with python.
 - [deepsport-utilities](https://gitlab.com/deepsport/deepsport_utilities): the toolkit for the datasets published during the deepsport project.

# Installation
Clone and install the repository with
```bash
git clone https://github.com/gabriel-vanzandycke/deepsport.git
cd deepsport
conda create --name deepsport python=3.8   # optional 
pip install -e .
```

Setup your environment by copying `.env.template` to `.env` and set:
- `DATA_PATH` to the list of folders to find datasets or configuration files, ordered by lookup priority.
- `RESULTS_FOLDER` to the full path to a folder in which outputs will be written (weigths and metrics).

# Dataset



# Training
```
python -m experimentator configs/ballsize.py --epochs 101 --kwargs "eval_epochs=range(0,101,20)"
```


# Evaluation
comming soon
