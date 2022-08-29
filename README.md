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

# Tasks

## Ball size estimation
This task uses a pre-processed dataset built from the `basketball-instants-dataset` with the following script:
```bash
python deepsport/scripts/prepare_ball_views_dataset.py --dataset-folder basketball-instants-dataset
```

The file generated (`basketball-instants-dataset/ball_views.pickle`) is an `mlworkflow.PickledDataset` whose items have the following attributes:
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

### Dataset splits

This repository uses the split defined by [`DeepSportDatasetSplitter`](https://gitlab.com/deepsport/deepsport_utilities/-/blob/main/deepsport_utilities/ds/instants_dataset/dataset_splitters.py#L6) which
1. Uses images from `KS-FR-CAEN`, `KS-FR-LIMOGES` and `KS-FR-ROANNE` arenas for the **testing-set**.
2. Randomly samples 15% of the remaining images for the **validation-set**
3. Uses the remaining images for the **training-set**.

The **testing-set** should not be used except to evaluate your model and when communicating about your method.


# Training
You can launch a new training by running the following command. Be sure the command is ran from the project root folder or add it to your `DATA_PATH` environment.
```
python -m experimentator configs/ballsize.py --epochs 101 --kwargs "eval_epochs=range(0,101,20)"
```

