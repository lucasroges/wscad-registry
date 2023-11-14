# WSCAD - Registry

This repository includes the datasets and algorithms to perform the experiments made in the paper "Dynamic Provisioning of Container Registries in Edge Computing Infrastructures" submitted to the *XXIV Simpósio em Sistemas Computacionais de Alto Desempenho* (WSCAD). 

## Configuring the environment

This project was configured with Poetry to manage its dependencies. Please ensure you have Python 3.10+ and Poetry installed. If not, you can install Poetry following the instructions [here](https://python-poetry.org/docs/#installation).

After installing these tools, you can install the dependencies of the project with the following command:


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

To run the experiments, navigate to the repository root folder and run the following command:

```sh
poetry run python run_experiments.py
```

Although the script runs a few cases in parallel, we recommend to run the script in background because it might take a while to complete all the executions.

## Getting the results

After the executions finished, the logs are available at the `logs` directory. To analyze the results, run the `analysis.ipynb` notebook. You can modify this notebook to explore different metrics and datasets for various scenarios.
