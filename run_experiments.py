# Importing Python libraries
from subprocess import Popen, DEVNULL, TimeoutExpired

import itertools
import os

NUMBER_OF_PARALLEL_PROCESSES =  os.cpu_count() - 1

SEED = "1"

def run_simulation(dataset: str):
    """Executes the simulation with the specified parameters.

    Args:
        dataset (tuple): Tuple containing the path to the dataset file, the algorithm name and a boolean indicating if some mapped servers should not host container registries. 
    """
    cmd = f"poetry run python -m simulation -a {dataset[1]} -d {dataset[0]} -s {SEED} -n 3600"
    print(f"    cmd = {cmd}")

    return Popen(cmd.split(" "), stdout=DEVNULL, stderr=DEVNULL)
    #return Popen(cmd.split(" "), stdout=open(f"logs/seed_{seed}.log", "w"), stderr=DEVNULL)


# Parameters
datasets = [
    ("datasets/central.json", "central"),
    ("datasets/community.json", "community"),
    ("datasets/p2p.json", "p2p"),
    ("datasets/p2p.json", "dynamic"),
]

print(f"Datasets: {datasets}")
print()

# Generating list of combinations with the parameters specified
combinations = list(
    itertools.product(
        datasets
    )
)

processes = []

# Executing simulations and collecting results
print(f"EXECUTING {len(combinations)} COMBINATIONS")
for i, parameters in enumerate(combinations, 1):
    # Parsing parameters
    dataset = parameters[0]

    print(f"\t[Execution {i}]")
    print(f"\t\t[dataset={dataset}]")

    # Executing algorithm
    proc = run_simulation(
        dataset=dataset,
    )

    processes.append(proc)

    while len(processes) >= NUMBER_OF_PARALLEL_PROCESSES:
        for proc in processes:
            try:
                proc.wait(timeout=1)

            except TimeoutExpired:
                pass

            else:
                processes.remove(proc)
                print(f"PID {proc.pid} finished")

    print(f"{len(processes)} processes running in parallel")