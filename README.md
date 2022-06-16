# CommunicativeOASYS
Decision-Theoretic Planning with Communication in Open Multiagent Systems (UAI 2022)

This project contains the source code for the domain, planning algorithms, and simulations used for the UAI 2022 paper listed above, implemented in the Cython programming language (similar to Python, but compiles into faster C code).  The repository contains several folders:

- `oasys/planning`: the planning algorithms considered in the study
- `oasys/domains/wildfire`: the implementation of Wildfire Suppression domain and its simulator for running experiments
- `oasys/agents`, `oasys/domains`, `oasys/simulation`, and `oasys/structures`: the generic classes used to implement the project, which can be extended with new child classes to represent other domains

## Compilation

To compile the source code, run the following from the main directory of the repository:

``` 
python setup.py build_ext --inplace
```

## Setups 

The study considered three environment setups described in paper, numbered setups 1-3.  The parameters defining each setup can be found in `oasys/domains/wildfire/wildfire_settings.pyx`.  In the code, the setups have been renumbered 10X, 20X, and 30X (respectively), where the last digit X defines the amount of communication cost:

- X=0: zero communication cost
- X=1: 0.05 cost per sent message
- X=2: 0.1 cost per sent message
- X=3: 0.2 cost per sent message
- X=4: 0.5 cost per sent message
- X=5: 1.0 cost per sent message

## Generating Level 0 Policies Offline

The `scripts/wildfire/generate_wildfire_pomcppf_policy.py` and `scripts/wildfire/generate_wildfire_pomcppfcomm_policy.py` programs can be used to generate level 0 policies offline to speed up level 1 planning, although this is not necessary.  The parameters to these programs are:

```
python generate_wildfire_pomcppf_policy.py <setup> 0.0 <trajectories> <horizon> <ucb_c> <number of particles> 1 <processes>
```

and


```
python generate_wildfire_pomcppfcomm_policy.py <setup> 0.0 <trajectories> <horizon> <ucb_c> <number of particles> 1 <processes>
```

which will use multi-processing with the given number of `<processes>` to generate policies for all agents in a given setup using the hyperparameters to the planning algorithms specified.  The offline policies computed for the study are provided in the offline_policies folder.

## Running Experiments

The `scripts/wildfire/run_wildfire_ipomcppf_simulation.py` and `scripts/wildfire_ipomcppfcomm_simulation.py` programs can be used to run the experiments for the I-POMCP-PF and CI-POMCP-PF algorithms, respectively.  The paramters to these programs are:

```
python run_wildfire_ipomcppf_simulation.py <setup> <start run> <stop run> 0.0 <trajectories> <horizon> <ucb_c> <planning level> <number of particles> 1 <1 to use offline policies for level 0, else 0 for fully online planning>
```

and


```
python run_wildfire_ipomcppfcomm_simulation.py <setup> <start run> <stop run> 0.0 <trajectories> <horizon> <ucb_c> <planning level> <number of particles> 1 <1 to use offline policies for level 0, else 0 for fully online planning>
```

where `<start run>` and `<stop run>` define the range of unique run numbers that will be executed by the simulator (each defining a unique random seed for the start of a simulation run).  If offline level 0 policies are used, the file containing the policy should be in the current directory from where the simulation program is run.
