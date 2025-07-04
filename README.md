# Subspace-constrained randomized coordinate descent (SC-RCD)

This repository contains the Python code used for the numerical experiments in the paper [Subspace-constrained randomized coordinate descent for linear systems with good low-rank matrix approximations](https://arxiv.org/abs/2506.09394) by Jackie Lok and Elizaveta Rebrova.

## Algorithm overview

The SC-RCD algorithm constrains the dynamics of (block) randomized coordinate descent – a simple, lightweight iterative solver – within a particular subspace corresponding to an efficiently computable low-rank matrix approximation such as RPCholesky.

For solving a $n \times n$ positive semidefinite (psd) linear system $Ax = b$, the iterations remain lightweight, requiring $O(dn)$ extra arithmetic per iteration given a rank $d$ approximation $\widehat{A} = F F^{T}$ in factorized form.
If the coordinates are simply sampled with probability proportional to the residual matrix $A^{\circ} = A - \widehat{A}$ in each iteration, then, compared to the classic randomized coordinate descent method, the convergence rate improves from a dependence on the spectrum of $A$ to a dependence on the spectrum of $A^{\circ}$.
This makes the SC-RCD method particularly effective if the matrix $A$ has rapid spectral decay.

## Dependencies

The `RPCholesky` directory contains the implementation of the RPCholesky algorithm from the Github repository https://github.com/eepperly/Randomly-Pivoted-Cholesky/ accompanying the papers
- [Randomly pivoted Cholesky: Practical approximation of a kernel matrix with few entry evaluations](https://arxiv.org/abs/2207.06503) by Yifan Chen, Ethan N. Epperly, Joel A. Tropp, and Robert J. Webber; and
- [Embrace rejection: Kernel matrix approximation by accelerated randomly pivoted Cholesky](https://arxiv.org/abs/2410.03969) by Ethan N. Epperly, Joel A. Tropp, and Robert J. Webber.

The `kaczmarz-plusplus` directory contains the implementation of the CD++ algorithm from the Github repository https://github.com/EdwinYang7/kaczmarz-plusplus accompanying the paper
- [Randomized Kaczmarz Methods with Beyond-Krylov Convergence](https://arxiv.org/abs/2501.11673) by Michał Dereziński, Deanna Needell, Elizaveta Rebrova, and Jiaming Yang.

## Data

We download a range of kernel ridge regression datasets from various sources considered in the paper 
- [Robust, randomized preconditioning for kernel ridge regression](https://arxiv.org/abs/2304.12465) by Mateo Díaz, Ethan N. Epperly, Zachary Frangella, Joel A. Tropp, and Robert J. Webber (see Table 1).

The script to download the datasets into the `/data` directory, which can be obtained from the accompanying Github repository https://github.com/eepperly/Robust-randomized-preconditioning-for-kernel-ridge-regression/, can be run using the following command:
```bash
$ cd data
$ python download_data.py
```

## Running SC-RCD experiments

The experiments can be performed by running the Python code located in the `/experiments` folder, with the desired parameters specified inside the file.

- The files `run_test.py`, `run_test_varyl.py`, `run_test_varyd.py`, `run_test_spectrum.py` are used for experiments solving kernel ridge regression (KRR) problems. In particular, implicit representation of the kernel matrix is supported, where entries of the input matrix can be directly evaluated from the input features as needed (i.e., it does not need to be entirely stored in memory).
- The files `run_test_psd.py`, `run_test_psd_varyl.py`, `run_test_psd_varyd.py`, `run_test_psd_spectrum.py` are used for experiments solving general psd linear systems.
- The Python files with the prefix `plot_` are used to plot the outputs of the corresponding test files.

**Example:**
```bash
$ python run_test.py
$ python plot_run_test.py
```

To run the experiments on different dataset, the `selected_dataset` argument can be changed to the id of a downloaded dataset. A list of valid datasets is stored in the `datasets` and `datasets_psd` dictionaries in the file `test_helper.py`.

Each experiment can also be performed by running the corresponding shell script, which both runs the test and produces the figure. The parameters are specified as keyword arguments to the Python files that are called inside the scripts. This makes it convenient to run the same experiment over different parameters (e.g., `selected_dataset`, `l`, `d`) or to average over a number of samples (possibly with parallelization).

**Example:**
```bash
$ bash run_test.sh
```

## Reproducing experiments in the paper

Shell scripts to reproduce the experiments in the paper with the specific parameters used can be found in the following folders:
- `/experiments/experiment_psd_simlowrank/`
- `/experiments/experiment_krr_lhc/`
- `/experiments/experiment_krr_sensorless/`
- `/experiments/experiment_krr_app/`

**Example:**
```bash
cd experiments/experiment_psd_simlowrank
bash run_test_psd.sh
bash run_test_psd_spectrum.sh
bash run_test_psd_varyl.sh
bash run_test_psd_varyd.sh
```