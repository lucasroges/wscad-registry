# Importing Python libraries
from subprocess import Popen, DEVNULL, TimeoutExpired

import itertools
import os

NUMBER_OF_PARALLEL_PROCESSES =  os.cpu_count() - 1

SEED = "1"

def run_simulation(algorithm, dataset):
    """Executes the simulation with the specified parameters.

    Args:
        algorithm (str): Algorithm to be executed.
        dataset (str): Dataset to be used.
    """
    cmd = f"poetry run python -m simulation -a {algorithm} -d {dataset} -s {SEED} -n 3600"
    print(f"    cmd = {cmd}")

    return Popen([cmd], stdout=DEVNULL, stderr=DEVNULL, shell=True)


# Parameters
algorithms = [
    ("central", "central"),
    ("community", "community"),
    ("p2p", "p2p"),
    ("dynamic", "p2p"),
    ("dynamic_enhanced", "p2p"),
]

number_of_nodes = [
    100,
    196
]

print(f"GENERATING {len(algorithms) * len(number_of_nodes)} COMBINATIONS")

# Generating list of combinations with the parameters specified
combinations = list(
    itertools.product(
        algorithms,
        number_of_nodes
    )
)

processes = []

# Executing simulations and collecting results
print(f"EXECUTING {len(combinations)} COMBINATIONS")
for i, parameters in enumerate(combinations, 1):
    # Parsing parameters
    algorithm = parameters[0]
    number_of_nodes = parameters[1]

    print(f"\t[Execution {i}]")
    print(f"\t\t[algorithm={algorithm[0]}] [number_of_nodes={number_of_nodes}]")

    # Executing algorithm
    proc = run_simulation(
        algorithm=algorithm[0],
        dataset=f"datasets/{algorithm[1]}\;nodes\={number_of_nodes}.json"
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
