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

Setup your environment by copying `.env.template` to `.env` and set the following variables:
- `LOCAL_STORAGE` to the folder in which downloaded datasets should be stored.
- `DATA_PATH` to the list of folders to find datasets or configuration files, ordered by lookup priority.
- `RESULTS_FOLDER` to the full path to a folder in which outputs will be written (weigths and metrics).

**Note:** Environment variable like `${HOME}` must be enclosed in curly brackets.

# Datasets

The different tasks rely on datasets to train and evaluate models. To download them programatically, you need the kaggle CLI:

```bash
pip install kaggle
```

Go to your Kaggle Account settings page and click on `Create new API Token` to download the file to be saved as
`~/.kaggle/kaggle.json` for authentication.


## Basketball-Instants-Dataset

The [Basketball-Instants-Dataset](https://www.kaggle.com/datasets/deepsportradar/basketball-instants-dataset) is a
dataset of independant instants (raw images captured at the same instant by the Keemotion automated production system).

The dataset can be downloaded and unzipped manually in the `basketball-instants-dataset/` folder of the project, or
programmatically with:

```bash
kaggle datasets download deepsportradar/basketball-instants-dataset
unzip -qo ./basketball-instants-dataset.zip -d basketball-instants-dataset
```

## Ballistic-Raw-Sequences-Dataset

The [Ballistic-Raw-Sequences-Dataset](https://www.kaggle.com/datasets/gabrielvanzandycke/ballistic-raw-sequences) is a
dataset of raw sequences captured by the Keemotion production system.

The dataset can be downloaded and unzipped manually in the `ballistic-raw-sequences/` folder of the project, or
programmatically with:

```bash
kaggle datasets download gabrielvanzandycke/ballistic-raw-sequences
unzip -qo ./ballistic-raw-sequences.zip -d ballistic-raw-sequences
```

# Tasks

The tasks are determined by a configuration file (located in the `configs` folder) that uses several functions and objects defined in the `models` and `tasks` folders. The tasks rely on a pre-processed dataset that needs to be computed and stored in your `DATA_PATH`.

The models can be trained by running the following command from the project root folder (or by adding the project root folder to the `DATA_PATH` environment variable):
```bash
python -m experimentator configs/<config-file> --epochs <numper-of-epochs>
```

Note: Configuration parameters can be overwritten from the command line by adding `--kwargs "<param-name>=<param-value>"`

| Task name    | Configuration file    | Dataset generation script                 | Notebook available |
|--------------|-----------------------|-------------------------------------------|:------------------:|
| **BallSeg**  | `configs/ballseg.py`  | `scripts/prepare_camera_views_dataset.py` |         yes        |
| **PIFBall**  | `configs/pifball.py`  | `scripts/prepare_camera_views_dataset.py` |         yes        |
| **BallSize** | `configs/ballsize.py` | `scripts/prepare_ball_views_dataset.py`   |         yes        |

## BallSeg: ball detection with a segmentation approach

This tasks addresses ball detection in basketball scenes. The pre-processed dataset items have the following attributes:
- `image`: a `numpy.ndarray` RGB image with ball visible somewhere.
- `calib`: a [`calib3d.Calib`](https://ispgroupucl.github.io/calib3d/calib3d/calib.html#implementation) object describing the calibration data associated to `image` using the [Keemotion convention](https://gitlab.com/deepsport/deepsport_utilities/-/blob/main/calibration.md#working-with-calibrated-images-captured-by-the-keemotion-system).
- `ball` : a [`Ball`](https://gitlab.com/deepsport/deepsport_utilities/-/blob/main/deepsport_utilities/ds/instants_dataset/instants_dataset.py#L264) object with attributes:
  - `center`: the ball 3D position as a [`calib3d.Point3D`](https://ispgroupucl.github.io/calib3d/calib3d/points.html) object (use `calib.project_3D_to_2D(ball.center)` to retrieve pixel coordinates).
  - `visible`: a flag telling if ball is visible (always `True` in this file)

The notebook to load the dataset and run the model training is available here: `notebooks/run_ballseg_experiment.ipynb`.

## PIFBall: ball detection with a Part-Intensity-Field

This tasks addresses ball detection using a keypoint detection approach greatly inspired by [PifPaf](https://openaccess.thecvf.com/content_CVPR_2019/papers/Kreiss_PifPaf_Composite_Fields_for_Human_Pose_Estimation_CVPR_2019_paper.pdf). The pre-processed dataset is the same than the one from the _BallSeg_ task.

**Important note:** The computation required for the evaluation phase is extremely long when the model is untrained. For this reasons, evaluation phase should be skipped for the first few epochs:
```bash
python -m experimentator configs/pifball.py --epochs 101 --kwargs "eval_epochs=range(20,101,20)"
```

## BallSize: ball diameter estimation for 3D localization

This tasks addresses ball 3D localization from a single calibrated image. The pre-processed dataset items have the following attributes:
- `image`: a `numpy.ndarray` RGB image thumbnail centered on the ball.
- `calib`: a [`calib3d.Calib`](https://ispgroupucl.github.io/calib3d/calib3d/calib.html#implementation) object describing the calibration data associated to `image` using the [Keemotion convention](https://gitlab.com/deepsport/deepsport_utilities/-/blob/main/calibration.md#working-with-calibrated-images-captured-by-the-keemotion-system).
- `ball` : a [`Ball`](https://gitlab.com/deepsport/deepsport_utilities/-/blob/main/deepsport_utilities/ds/instants_dataset/instants_dataset.py#L264) object with attributes:
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

## Ball state classification

To be released


# License

This project is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License (CC BY-NC-ND 4.0) - visit [https://creativecommons.org/licenses/by-nc-nd/4.0/](https://creativecommons.org/licenses/by-nc-nd/4.0/) for more details.


# Citation

If you find this work useful and want to cite it in your own research, projects, or publications, please use the following citation.

Please make sure to comply with the terms of the license when using or referencing this work.

If you have any questions about citing the work or need further information, you should contact me.


## Datasets

- Initial DeepSport dataset (315 samples):
```
@inproceedings{VanZandycke2019,
  Author = {Gabriel {Van Zandycke} and Christophe {De Vleeschouwer}},
  Title = {Real-time CNN-based segmentation architecture for ball detection in a single view setup},
  Booktitle = {MMSports 2019 - Proceedings of the 2nd International Workshop on Multimedia Content Analysis in Sports, co-located with MM 2019},
  Year = {2019}
  DOI = {10.1145/3347318.3355517},
}
```

- Extended DeepSport dataset (364 samples):
```
@inproceedings{deepsportradar,
  Author = {Gabriel {Van Zandycke} and Vladimir Somers and Maxime Istasse and Carlo {Del Don} and Davide Zambrano},
  Title = {DeepSportradar-v1: Computer Vision Dataset for Sports Understanding with High Quality Annotations},
  Booktitle = {Proceedings of the 5th International ACM Workshop on Multimedia Content Analysis in Sports},
  Year = {2022},
  DOI = {10.1145/3552437.3555699},
}
```

- Ball 3D localization over ballistic trajectories (233 samples):
```
@inproceedings{ball3d,
  Author = {Gabriel {Van Zandycke} and Christophe {De Vleeschouwer}},
  Title = {3D Ball Localization From A Single Calibrated Image},
  Booktitle = {IEEE Computer Society Conference on Computer Vision and Pattern Recognition Workshops},
  year = {2022}
  DOI = {10.1109/CVPRW56347.2022.00391},
}
```

## Tasks

- Ball detection with a segmentation approach:

```
@inproceedings{VanZandycke2019,
  Author = {Gabriel {Van Zandycke} and Christophe {De Vleeschouwer}},
  Title = {Real-time CNN-based segmentation architecture for ball detection in a single view setup},
  Booktitle = {MMSports 2019 - Proceedings of the 2nd International Workshop on Multimedia Content Analysis in Sports, co-located with MM 2019},
  Year = {2019}
  DOI = {10.1145/3347318.3355517},
}
```

- Ball detection with a keypoint detection approach:

```
@inproceedings{Ghasemzadeh2021,
  Author = {Abolfazl {S.A. Ghasemzadeh} and Gabriel {Van Zandycke} and Maxime Istasse and Niels Sayez and Amirafshar Moshtaghpour and Christophe {De Vleeschouwer}},
  Booktitle = {32nd British Machine Vision Conference 2021 (BMVC 2021)},
  Title = {DeepSportLab: a Unified Framework for Ball Detection, Player Instance Segmentation and Pose Estimation in Team Sports Scenes},
  Year = {2021}
}
```

- Ball 3D localization with a diameter approach method:

```
@inproceedings{ball3d,
  Author = {Gabriel {Van Zandycke} and Christophe {De Vleeschouwer}},
  Title = {3D Ball Localization From A Single Calibrated Image},
  Booktitle = {IEEE Computer Society Conference on Computer Vision and Pattern Recognition Workshops},
  year = {2022}
  DOI = {10.1109/CVPRW56347.2022.00391},
}
```

- Ball 3D localization baseline

```
@inproceedings{deepsportradar,
  Author = {Gabriel {Van Zandycke} and Vladimir Somers and Maxime Istasse and Carlo {Del Don} and Davide Zambrano},
  Title = {DeepSportradar-v1: Computer Vision Dataset for Sports Understanding with High Quality Annotations},
  Booktitle = {Proceedings of the 5th International ACM Workshop on Multimedia Content Analysis in Sports},
  Year = {2022},
  DOI = {10.1145/3552437.3555699},
}
```

- Ball 3D localization from a single image with a diameter approach:

```
comming soon
```

