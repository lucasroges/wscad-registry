# Importing Python libraries
from subprocess import Popen, DEVNULL, TimeoutExpired

import itertools
import os

NUMBER_OF_PARALLEL_PROCESSES =  os.cpu_count() - 1

def run_simulation(dataset: str, seed: int):
    """Executes the simulation with the specified parameters.

    Args:
        dataset (tuple): Tuple containing the path to the dataset file, the algorithm name and a boolean indicating if some mapped servers should not host container registries. 
        seed (int): Seed value for EdgeSimPy.
    """
    cmd = f"/home/dellcloud/.cache/pypoetry/virtualenvs/p2p-enhanced-registry-provisioning-7g8CcTQ_-py3.10/bin/python -m simulation -a {dataset[1]} -d {dataset[0]} -s {seed} -n 3600"
    print(f"    cmd = {cmd}")

    return Popen(cmd.split(" "), stdout=DEVNULL, stderr=DEVNULL)
    #return Popen(cmd.split(" "), stdout=open(f"logs/seed_{seed}.log", "w"), stderr=DEVNULL)


# Parameters
datasets = [
    ("datasets/central.json", "central"),
    ("datasets/community.json", "community"),
    ("datasets/p2p.json", "p2p"),
    ("datasets/p2p.json", "p2p_enhanced"),
]

seeds = [1, 2, 3, 4, 5]

print(f"Datasets: {datasets}")
print(f"Seeds: {seeds}")
print()

# Generating list of combinations with the parameters specified
combinations = list(
    itertools.product(
        datasets,
        seeds,
    )
)

processes = []

# Executing simulations and collecting results
print(f"EXECUTING {len(combinations)} COMBINATIONS")
for i, parameters in enumerate(combinations, 1):
    # Parsing parameters
    dataset = parameters[0]
    seed = parameters[1]

    print(f"\t[Execution {i}]")
    print(f"\t\t[dataset={dataset}] seed={seed}.")

    # Executing algorithm
    proc = run_simulation(
        dataset=dataset,
        seed=seed,
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