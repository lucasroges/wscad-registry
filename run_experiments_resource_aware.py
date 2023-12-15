# Importing Python libraries
from subprocess import Popen, DEVNULL, TimeoutExpired

import itertools
import os

NUMBER_OF_PARALLEL_PROCESSES =  os.cpu_count() - 1

SEED = "1"

def run_simulation(algorithm, dataset, replicas, percentage_of_replicated_images):
    """Executes the simulation with the specified parameters.

    Args:
        algorithm (str): Algorithm to be executed.
        dataset (str): Dataset to be used.
        replicas (int): Number of replicas to be used.
        percentage_of_replicated_images (float): Percentage of replicated images to be used.
    """
    cmd = f"poetry run python -m simulation -a {algorithm} -d {dataset} -s {SEED} -n 3600 -r {replicas} -p {percentage_of_replicated_images}"
    print(f"    cmd = {cmd}")

    return Popen([cmd], stdout=DEVNULL, stderr=DEVNULL, shell=True)


# Parameters
algorithms = [
    ("resource_aware_dynamic", "p2p"),
]

number_of_nodes = [
    100,
    196
]

mobility = [
    "faster",
    "slower"
]

replicas = [1, 2, 3]

percentage_of_replicated_images = [0.5, 0.6, 0.7, 0.8, 0.9, 1]

print(f"GENERATING {len(algorithms) * len(number_of_nodes) * len(mobility) * len(replicas) * len(percentage_of_replicated_images)} COMBINATIONS")

# Generating list of combinations with the parameters specified
combinations = list(
    itertools.product(
        algorithms,
        number_of_nodes,
        mobility,
        replicas,
        percentage_of_replicated_images
    )
)

processes = []

# Executing simulations and collecting results
print(f"EXECUTING {len(combinations)} COMBINATIONS")
for i, parameters in enumerate(combinations, 1):
    # Parsing parameters
    algorithm = parameters[0]
    number_of_nodes = parameters[1]
    mobility = parameters[2]
    replicas = parameters[3]
    percentage_of_replicated_images = parameters[4]

    print(f"\t[Execution {i}]")
    print(f"\t\t[algorithm={algorithm[0]}] [number_of_nodes={number_of_nodes}] [mobility={mobility}]")

    # Executing algorithm
    proc = run_simulation(
        algorithm=algorithm[0],
        dataset=f"datasets/{algorithm[1]}\;nodes\={number_of_nodes}\;mobility\={mobility}.json",
        replicas=replicas,
        percentage_of_replicated_images=percentage_of_replicated_images
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
