# Developping simulation and statistical evaluation tools for a non-linear mixed-effect models

The code used for tests and communication during my internship at Paris Brain Institut (ICM) - INRIA ARAMIS Laboratory on the software [leaspy](https://github.com/aramis-lab/leaspy). The associated pull request is [Add Simulation Algo for Joint Model](https://github.com/aramis-lab/leaspy/pull/499). 

## Features 
In the leaspy branch associated with this pull request, you can find:
- Implementation of a simulation algorithm for the joint model ```JointSimulationAlgorithm``` which inherits from the ```SimulationAlgorithm``` class of leaspy.
- Unit tests for the joint simulation algorithm. These tests follow the same structure as the existing tests in leaspy using pytest to ensure the correctness of the implementation. 

In addition, a set of notebooks is provided in this repository to test the algorithm and evaluate the metrics used to compare the simulated data to the real data. These notebooks are intended to be used for communication and demonstration purposes, and they can be adapted for further testing and evaluation of the joint simulation algorithm.

## Notebooks

| # | Name | Description |
|---|------|-------------|
| 1 | [Base test](notebooks/1.%20Base%20test.ipynb) | Basic test of the joint simulation algorithm. |
| 2 | [Metrics](notebooks/2.%20Metrics.ipynb) | Evaluation of the metrics used to compare the simulated data to the real data. |
| 3 | [Paper's Pipeline](notebooks/3.%20Paper's%20Pipeline.ipynb) | Reproduction of the metrics used in the original paper (Ortholand et al., 2025) to evaluate the simulated parameters. |

## Author
- **Jean-Vincent Martini**