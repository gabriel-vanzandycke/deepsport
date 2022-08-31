This repository holds the task specific software made public for my PhD under the DeepSport project funded by the Walloon Region of Belgium, Keemotion and SportRadar.
It goes along with several open-source libraries that I developed during that time:
 - [calib3d](https://github.com/ispgroupucl/calib3d): a library to ease computations in 2D and 3D homogenous coordinates and projective geometry with camera calibration.
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

# Datasets

The different tasks rely on datasets to train and evaluate models.

## Basketball-Instants-Dataset

This repository uses the [Basketball-Instants-Dataset](https://www.kaggle.com/datasets/deepsportradar/basketball-instants-dataset), a dataset of raw images captured at the same instant from the Keemotion production system.

The dataset can be downloaded and unzipped manually in the `basketball-instants-dataset/` folder of the project. To do it programmatically, you need the kaggle CLI:

```bash
pip install kaggle
```

Go to your Kaggle Account settings page and click on `Create new API Token` to download the file to be saved as `~/.kaggle/kaggle.json` for authentication. Finally download and unzip the dataset with:

```bash
kaggle datasets download deepsportradar/basketball-instants-dataset
unzip -qo ./basketball-instants-dataset.zip -d basketball-instants-dataset
```

# Tasks

The tasks are defined by a configuration file (located in the `configs` folder) that uses several functions and objects defined in the `models` and `tasks` folders.
Each task relies on a pre-processed dataset in the form of an `mlworkflow.PickledDataset` that must be computed and stored in the `basketball-instants-dataset` folder.

The models can be trained by running the following command from the project root folder (or by adding the project root folder to the `DATA_PATH` environment variable):
```bash
python -m experimentator configs/<config-file> --epochs <numper-of-epochs>
```

## BallSeg: ball detection with a segmentation approach
This tasks addresses ball detection in basketball scenes. The pre-processed dataset can be built with the `deepsport/scripts/prepare_camera_views_dataset.py` script. Its items have the following attributes:
- `image`: a `numpy.ndarray` RGB image with ball visible somewhere.
- `calib`: a [`calib3d.Calib`](https://ispgroupucl.github.io/calib3d/calib3d/calib.html#implementation) object describing the calibration data associated to `image` using the [Keemotion convention](https://gitlab.com/deepsport/deepsport_utilities/-/blob/main/calibration.md#working-with-calibrated-images-captured-by-the-keemotion-system).
- `ball` : a [`BallAnnotation`](https://gitlab.com/deepsport/deepsport_utilities/-/blob/main/deepsport_utilities/ds/instants_dataset/instants_dataset.py#L264) object with attributes:
  - `center`: the ball 3D position as a [`calib3d.Point3D`](https://ispgroupucl.github.io/calib3d/calib3d/points.html) object (use `calib.project_3D_to_2D(ball.center)` to retrieve pixel coordinates).
  - `visible`: a flag telling if ball is visible (always `True` in this file)

The configuration file is `configs/ballseg.py` and the model can be trained by running the following command from the project root folder.
```bash
python -m experimentator configs/ballseg.py --epochs 101 --kwargs "eval_epochs=range(0,101,20)"
```


## Ball size estimation

This tasks addresses ball 3D localization from a single calibrated image. The pre-processed dataset can be built with the `deepsport/scripts/prepare_ball_views_dataset.py` script. Its items have the following attributes:
- `image`: a `numpy.ndarray` RGB image thumbnail centered on the ball.
- `calib`: a [`calib3d.Calib`](https://ispgroupucl.github.io/calib3d/calib3d/calib.html#implementation) object describing the calibration data associated to `image` using the [Keemotion convention](https://gitlab.com/deepsport/deepsport_utilities/-/blob/main/calibration.md#working-with-calibrated-images-captured-by-the-keemotion-system).
- `ball` : a [`BallAnnotation`](https://gitlab.com/deepsport/deepsport_utilities/-/blob/main/deepsport_utilities/ds/instants_dataset/instants_dataset.py#L264) object with attributes:
  - `center`: the ball 3D position as a [`calib3d.Point3D`](https://ispgroupucl.github.io/calib3d/calib3d/points.html) object (use `calib.project_3D_to_2D(ball.center)` to retrieve pixel coordinates).
  - `visible`: a flag telling if ball is visible.

You can visualize this dataset the following way:
```python
from mlworkflow import PickledDataset
from matplotlib import pyplot as plt
ds = PickledDataset("basketball-instants-dataset/ball_views.pickle")
for key in ds.keys:
    item = ds.query_item(key)
    plt.imshow(item.image)
    plt.title("ball size: {:.1f}".format(item.calib.compute_length2D(23, item.ball.center)[0]))
    plt.show()
    break # avoid looping through all dataset
```


This task uses the split defined by [`DeepSportDatasetSplitter`](https://gitlab.com/deepsport/deepsport_utilities/-/blob/main/deepsport_utilities/ds/instants_dataset/dataset_splitters.py#L6) which
1. Uses images from `KS-FR-CAEN`, `KS-FR-LIMOGES` and `KS-FR-ROANNE` arenas for the **testing-set**.
2. Randomly samples 15% of the remaining images for the **validation-set**
3. Uses the remaining images for the **training-set**.
The **testing-set** should not be used except to evaluate your model and when communicating about your method.


The configuration file is `configs/ballsize.py` and the model can be trained by running the following command.
```
python -m experimentator configs/ballsize.py --epochs 101 --kwargs "eval_epochs=range(0,101,20)"
```

