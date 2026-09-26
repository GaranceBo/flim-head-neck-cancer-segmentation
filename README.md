# flim-head-neck-cancer-segmentation
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22983044.svg)](https://doi.org/10.5281/zenodo.22983044)

# Deep learning segmentation of high resolution FLIM head and neck cancer images

This repository contains the source code used for the study:

**The role of spatial resolution in cellular-scale fluorescence lifetime imaging deep-learning segmentation of head and neck cancer cryosections**

The study investigates deep-learning-based segmentation of tumor tissue in fluorescence lifetime imaging microscopy (FLIM) images acquired from human head and neck cancer cryosections and xenograft mouse models cryosections.

The repository contains the code used for data preprocessing, model definition, training, inference, evaluation, and image reconstruction.

## Repository structure

```text
flim-head-neck-cancer-segmentation/
│
├── src/
│   ├── __init__.py
│   ├── models.py
│   ├── data.py
│   ├── losses.py
│   ├── training.py
│   ├── evaluation.py
│   └── utils.py
│
├── data/
│   └── README.md
│
├── outputs/
│   └── .gitkeep
│
├── main_human.py
├── main_xenograft.py
├── requirements.txt
├── CITATION.cff
└── README.md
```

### Source code

* `src/models.py` contains the neural-network architectures evaluated in the study.
* `src/data.py` contains dataset loading, preprocessing, augmentation, splitting, and PyTorch dataset utilities.
* `src/losses.py` contains the loss functions and Dice-related metrics used during training.
* `src/training.py` contains the model training procedure, including optimization, mixed-precision training, validation, and early stopping.
* `src/evaluation.py` contains model evaluation and segmentation performance metrics.
* `src/utils.py` contains reproducibility utilities, threshold selection, visualization functions, reconstruction plotting, and additional model/data analyses.
* `main_human.py` contains the example workflow for the human tumor dataset.
* `main_xenograft.py` contains the example workflow for the xenograft dataset.

## Data

The imaging datasets are **not stored in this GitHub repository**.

The complete datasets associated with the study are available through Figshare:

**10.6084/m9.figshare.34003191**

The Figshare repository contains the human tumor and xenograft datasets, reconstructed images, histopathological images and associated metadata.

Please refer to the Figshare `DATA_DESCRIPTION.md` file for detailed information about the structure, dimensions, data types, channel ordering, normalization, and interpretation of the arrays.

### Downloading the data

After downloading the required datasets from Figshare, place the files in the appropriate local data directory.

For example:

```text
data/
├── human/
│   ├── Tu2_tiles.npy
│   ├── Tu3_tiles.npy
│   ├── ...
│
└── xenograft/
    └── ...
```

The large numerical datasets are intentionally excluded from this repository.

## Installation

### Using pip

Create a Python environment and install the required dependencies:

```bash
pip install -r requirements.txt
```

The experiments were performed using PyTorch with CUDA acceleration. GPU availability is needed for reproducing the training experiments.

## Running the examples

After downloading the required datasets and installing the dependencies, the example workflows can be run from the repository root.

For the human dataset:

```bash
python main_human.py
```

For the xenograft dataset:

```bash
python main_xenograft.py
```

The scripts illustrate the complete workflow:

1. Loading the imaging data.
2. Preprocessing and dataset preparation.
3. Creating the neural-network model.
4. Training with the experimental training procedure.
5. Plotting the training history.
6. Selecting the segmentation threshold using the validation set.
7. Evaluating the model on the test set.
8. Generating reconstructed segmentation images.

Note: All training experiments use by default a training/validation split of 80/20. This split was also used for all experiments conducted for the publication. Splits can however be modified at will by the user.

## Models

The repository contains the neural-network architectures evaluated in the study, including convolutional, transformer-based, and hybrid segmentation architectures.

All model implementations used for the analyses reported in the publication are included in `src/models.py`, and further analyzed in the associated publication.

## Reproducibility

The repository provides the code required to reproduce the preprocessing, training, inference, evaluation, and reconstruction procedures used in the study.

Random seeds can be explicitly set using the utilities provided in `src/utils.py`.

The exact experimental settings used in the study, including image resolution, dataset splits, training parameters, and evaluation procedures, are documented in the associated publication.

## Outputs

Generated model weights, training plots, evaluation results, and reconstructed images are not stored in the GitHub repository by default.

They can be generated locally by running the corresponding example scripts, or writing derived scripts using the functions provided.

## Citation

If you use this code, please cite the associated publication and this software repository.

The recommended software citation is provided in `CITATION.cff`.

The dataset should be cited separately using the DOI provided by Figshare.

### Associated publication

**The role of spatial resolution in cellular-scale fluorescence lifetime imaging deep-learning segmentation of head and neck cancer cryosections**

[Publication DOI / journal reference to be added]

### Dataset

The associated imaging dataset is available on Figshare:

**10.6084/m9.figshare.34003191**

### Software

An archived version of this repository is available through Zenodo:

**[10.5281/zenodo.22983044](https://doi.org/10.5281/zenodo.22983044)**

## License

The source code is distributed under the license specified in this repository.

The imaging dataset is distributed separately under the license specified in the associated Figshare record.

Please refer to the respective license files and repository records before redistributing the code or data.
