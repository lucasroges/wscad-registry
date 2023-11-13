# WSCAD - Registry

This repository includes the datasets and algorithms to perform the experiments made in the paper "Dynamic Provisioning of Container Registries in Edge Computing Infrastructures" submitted to the *XXIV Simpósio em Sistemas Computacionais de Alto Desempenho* (WSCAD). 

## Configuring the environment

```sh
poetry install
```

## Building the datasets

To build the necessary datasets to replicate the paper's experiments, run the following commands at the repository root folder:

```sh
poetry run python -m datasets -s 1 -i datasets/inputs/base_minimal.json -o central_minimal -rp central && \
poetry run python -m datasets -s 1 -i datasets/inputs/base_recommended.json -o central_recommended -rp central && \
poetry run python -m datasets -s 1 -i datasets/inputs/base_minimal.json -o community_minimal -rp community -c 6 && \
poetry run python -m datasets -s 1 -i datasets/inputs/base_recommended.json -o community_recommended -rp community -c 6 && \
poetry run python -m datasets -s 1 -i datasets/inputs/base_minimal.json -o p2p_minimal -rp p2p && \
poetry run python -m datasets -s 1 -i datasets/inputs/base_recommended.json -o p2p_recommended -rp p2p
```

## Running the experiments

To run the experiments, we only need to run the `run_experiments.py` file, which runs a few cases in parallel.

```sh
poetry run python run_experiments
```

We recommend to run the script in background, as it might take a while to complete all the executions.

## Getting the results

After the executions finished, the logs are available at the `logs` directory and we can run the `analysis.ipynb` as a whole to obtain the results shown in the paper.

This notebook can be modified to analyze different metrics.
