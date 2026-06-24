# FISTA-LD

This repository contains the code accompanying the paper *"A Proximal Gradient Method with Learned Deviations and Inertial Acceleration"*.

## Repository structure
- `code/`
    - `data.py`: Dataset generation (training, validation, and test).
    - `operators.py`: Operators used in the optimization problems.
    - `optimization.py`: Optimization algorithms.
    - `paper_experiments.py`: Reproduces experiments from the paper.
    - `paper_results.py`: Reproduces tables and figures from the paper.
    - `paths.py`: Utilities for managing paths.
    - `plots_and_metrics.py`: Plotting utilities and evaluation metrics.
    - `proxTV.py`: Inexact proximal operator for TV regularization.
    - `results.py`: Generates figures and tables from experimental results.
    - `single_experiment.py`: Runs a single training and testing experiment.
    - `traintest.py`: Training and testing routines.
- `ray_tracing/`: Fork of https://github.com/elias-helou/ray_tracing (commit 0f3a299), with a minor modification to the Makefile. Installed automatically as part of the setup procedure described below.
- `Dockerfile`: Creates the Docker image with all dependencies.

## Data

The experiments use data from the *AAPM Low Dose CT Grand Challenge Dataset* [[1]](#1)[[2]](#2).

In the `code/data/mayo_clinic_image/` subdirectory, extract `full_3mm.zip`, which can be downloaded from https://aapm.app.box.com/s/eaw4jddb53keg1bptavvvd1sf4x3pe9h/file/856956352254 (Accessed: 2026-05-31).

## Setup

The instructions below describe the environment used to run the experiments. The setup was tested on a Linux machine running Ubuntu 24.04.4 LTS.

### Prerequisites

- An NVIDIA GPU with at least 12 GB of VRAM
- At least 16 GB of RAM
- At least 40 GB of available disk space
- Docker: https://docs.docker.com/engine/install/ubuntu/
- NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

The experiments were conducted using Docker 29.5.3 and NVIDIA Container Toolkit 1.19.1.

### Running the container

From the repository root (`FISTA-LD/`), build the Docker image:
```
docker build --build-arg CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '.') -t fista-ld .
```

Then, run the container:
```
docker run --rm --runtime=nvidia --gpus '"device=0"' --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --mount type=bind,source=$(pwd)/code,target=/code -it fista-ld
```

This opens an interactive terminal inside the container, from which the experiments can be executed.

## Running experiments

### Example experiment

To run a training and testing experiment, run the command:
```
python single_experiment.py
```
The file `single_experiment.py` contains a simple training and testing experiment to verify that everything is working correctly. The first execution typically takes longer, since the datasets are generated during the first run. Edit `single_experiment.py` to choose different problems, algorithms, or training/testing protocols.

To generate tables and figures summarizing the experimental results, run the command below. The generated tables and figures are saved in the output directory indicated in the terminal. The tables are also printed in the terminal.
```
python results.py
```

### Reproducing paper results

To reproduce the experiments presented in the paper, run the following commands:
```
python paper_experiments.py
python paper_results.py
```
The generated results are saved in the output directory indicated in the terminal.

## Contact

For questions or comments, please contact: ggrillo@ime.usp.br.

## References
<a id="1">[1]</a> 
McCollough CH, Bartley AC, Carter RE, Chen B, Drees TA, Edwards P, Holmes DR 3rd, Huang AE, Khan F, Leng S, McMillan KL, Michalak GJ, Nunez KM, Yu L, Fletcher JG. Low-dose CT for the detection and classification of metastatic liver lesions: Results of the 2016 Low Dose CT Grand Challenge. Med Phys. 2017 Oct;44(10):e339-e352. doi: 10.1002/mp.12345. PMID: 29027235; PMCID: PMC5656004.\
<a id="2">[2]</a> https://www.aapm.org/grandchallenge/lowdosect/ (Accessed: 2026-05-31)