# Developping simulation and statistical evaluation tools for a non-linear mixed-effect models

The code used for tests and communication during my internship at Paris Brain Institut (ICM) - INRIA ARAMIS Laboratory on the software [leaspy](https://github.com/aramis-lab/leaspy). The associated pull request is [Add Simulation Algo for Joint Model](https://github.com/aramis-lab/leaspy/pull/499). This internship fulfills the academic requirements for the MVA Master’s program at ENS Paris-Saclay and CentraleSupélec.

## Features 
In the leaspy branch associated with this pull request, you can find:
- `JointSimulationAlgorithm` which inherits from `SimulationAlgorithm`.
- Added unit tests for `JointSimulationAlgorithm`.
- Adapted `estimate()` to work with the joint model and properly handle the events. It is used in `JointSimulationAlgorithm`.
- Added functional tests for this new `estimate`.

In addition, a set of notebooks is provided in this repository to test the algorithm and evaluate the metrics used to compare the simulated data to the real data. These notebooks are intended to be used for communication and demonstration purposes, and they can be adapted for further testing and evaluation of the joint simulation algorithm.

## Notebooks

| # | Name | Description |
|---|------|-------------|
| 1 | [Base test](notebooks/1.%20Base%20test.ipynb) | End-to-end demonstration of joint simulation: loads the built-in simulated dataset, fits a `JointModel`, runs `JointSimulationAlgorithm` to generate new patient trajectories and survival events, and visualizes the output with UMAP. |
| 2 | [Metrics](notebooks/2.%20Metrics.ipynb) | Statistical evaluation of simulation quality against the original reference dataset, using Wasserstein distance, Kolmogorov-Smirnov tests, Kaplan-Meier survival curves, visit-count distributions, and feature-trajectory correlations. Requires access to the original data. |
| 2a | [Wout data](notebooks/2a.%20Wout%20data.ipynb) | Same statistical evaluation as notebook 2, adapted for settings where the original reference data is unavailable. Assesses intrinsic properties of the simulated output (survival curves, visit distributions, trajectory density estimates) across multiple simulation runs. |
| 3 | [Paper's Pipeline](notebooks/3.%20Paper's%20Pipeline.ipynb) | Reproduces the simulation study from Ortholand et al. (2025). Repeatedly simulates datasets from a reference model, re-fits a fresh `JointModel` on each, and measures population-parameter and individual-level parameter recovery using REE, RRMSE, ICC, etc. |

## Project Structure

```
├── notebooks/                     # Jupyter notebooks (see table above)
│   ├── 1. Base test.ipynb
│   ├── 2. Metrics.ipynb
│   ├── 2a. Wout data.ipynb
│   └── 3. Paper's Pipeline.ipynb
├── deliverables/
├── notes/
└── README.md
```

A `cluster` branch is also available in the repository, containing code for parallelized simulation and fitting across multiple processes on a HPC cluster, as used in the paper's pipeline notebook.

## Reference

- Juliette Ortholand, Stanley Durrleman, and Sophie Tezenas du Montcel. A joint
spatiotemporal model for multiple longitudinal markers and competing events.
arXiv preprint arXiv:2501.08960, 2025.

## Author
- **Jean-Vincent Martini**