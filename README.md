# FISTA-LD

In this repository, we share the code for the article "A Proximal Gradient Method with learned deviations and inertial acceleration".

## Files
- /code/analyse_params.py -- Produces plots and tables to compare FISTA-LD with respect to different combinations of $\alpha$ and $\gamma$.
- /code/data.py -- Generates the datasets for training, validation, and testing.
- /code/operators.py -- Creates the operators considered in the minimization problems.
- /code/optimization.py -- Implementations of the optimization algorithms.
- /code/proxTV.py -- Implementation of the iterative algorithm to evaluate the proximal operator of the TV function inexactly.
- /code/results.py -- Produces plots and tables for the trained and tested algorithms.
- /code/results_lstv.py -- Same purpose as results.py, but with specific analysis for the TV-regularized least squares problem.
- /code/results_only_fista.py -- Same purpose as results.py, but with specific analysis for FISTA.
- /code/traintest.py --Training and testing the considered algorithms.
- Dockerfile -- Creates the Docker image with all dependencies.

### Ray tracing implementation

We used the ray tracing implementation from https://github.com/elias-helou/ray_tracing. The files 'binio.py', 'cuda_radon_torch.[o,so]', 'cuda_radon.[o,so]', 'pyraft.py', and 'ray_tracing_cuda_float_torch.py' in the 'code' directory are produced in the Makefile of the mentioned repository. You may need to compile these files on your machine for better compatibility. After that, you just need to replace the files present in this directory with the ones compiled on your machine.

### Data
For the proper behavior of the data generator, please follow the instructions in the directory '/code/data/mayo_clinic_image'.

## Setup

This is a suggested setup based on how we ran our experiments on our machine.

You will need to install Docker (see https://docs.docker.com/engine/install/ubuntu/) and the NVIDIA Container Toolkit (see https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

After that, inside the directory 'FISTA-LD', build the Docker image:
```
docker build -t fista-ld .
```

Then, run the image with GPU access (replace '/your/path/FISTA-LD' appropriately):
```
docker run --rm --runtime=nvidia --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --mount type=bind,source=/your/path/FISTA-LD/code,target=/code -it fista-ld
```

This will open a terminal in which you can run the implementations. For example, to train and test the selected algorithms, run the command:
```
python traintest.py
```
The file 'traintest.py' contains a simple training and testing experiment to verify that everything is working correctly. The first execution of the program will typically take more time (a few hours), since it builds the datasets. Edit 'traintest.py' after line 899 to choose different problems, algorithms, or training/testing protocols.

All tables generated in 'analyse_params.py', 'results.py', 'results_lstv.py', or 'results_only_fista.py' are printed in the terminal. The plots produced in these scripts are saved in the subdirectory 'plots'.

## Contact

Contact me by email: ggrillo@ime.usp.br.
