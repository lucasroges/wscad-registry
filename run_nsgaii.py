# Pymoo
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.util.display.column import Column
from pymoo.util.display.output import Output
from pymoo.core.problem import Problem
from pymoo.optimize import minimize

# Python libraries
from concurrent.futures import ProcessPoolExecutor, as_completed

import multiprocessing
import numpy as np
import random
import pickle
import json
import os

VERBOSE = False
PARALLEL = True
N_THREADS = multiprocessing.cpu_count()


#class MyOutput(Output):
#    """Creates a visualization on how the genetic algorithm is evolving throughout the generations."""
#    def __init__(self):
#        super().__init__()
#        self.x_mean = Column("x_mean", width=13)
#        self.x_std = Column("x_std", width=13)
#        self.columns += [self.x_mean, self.x_std]
#
#    def update(self, algorithm):
#        super().update(algorithm)
#        self.x_mean.set(np.mean(algorithm.pop.get("X")))
#        self.x_std.set(np.std(algorithm.pop.get("X")))


class RegistryProvisioningProblem(Problem):
    """Describes the registry provisioning as an optimization problem."""
    edge_servers_without_central_registry = 23
    algorithm_executions = 60

    def __init__(self, **kwargs):
        """Initializes the problem instance."""
        super().__init__(
            n_var=self.edge_servers_without_central_registry * self.algorithm_executions,
            n_obj=2,
            n_constr=1,
            xl=0,
            xu=1,
            type_var=int,
            **kwargs
        )

    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluates solutions according to the problem objectives.
        Args:
            x (list): Solution or set of solutions that solve the problem.
            out (dict): Output of the evaluation function.
        """
        if PARALLEL:
            executor = ProcessPoolExecutor(max_workers=N_THREADS)
            futures = [executor.submit(self.get_fitness_score_and_constraints, solution) for solution in x]
            output = [future.result() for future in as_completed(futures)]
            executor.shutdown()

        else:
            output = [self.get_fitness_score_and_constraints(solution=solution) for solution in x]

        out["F"] = np.array([item[0] for item in output])
        out["G"] = np.array([item[1] for item in output])

    def get_fitness_score_and_constraints(self, solution: list) -> tuple:
        """Calculates the fitness score and penalties of a solution based on the problem definition.
        Args:
            solution (list): Solution that solves the problem.
        Returns:
            tuple: Output of the evaluation function containing the fitness scores of the solution and its penalties.
        """
        # Run the simulation with the solution
        results = apply_solution(solution)

        # Gather the results
        output = evaluate_solution(results)

        return output

        


def genetic_algorithm(
    n_gen: int, pop_size: int, cross_prob: int, mutation_prob: int, seed: int = 1
) -> list:
    """Gets the allocation scheme used to drain servers during the maintenance using the genetic algorithm.
    Args:
        n_gen (int): Number of generations the genetic algorithm will run through.
        pop_size (int): Number of chromosomes that will represent the genetic algorithm's population.
        cross_prob (int): Crossover probability.
        mutation_prob (str): Mutation probability.
    Returns:
        [list]: Placement scheme returned by the genetic algorithm.
    """

    # Defining genetic algorithm's attributes
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=BinaryRandomSampling(),
        crossover=TwoPointCrossover(),
        mutation=BitflipMutation(),
        eliminate_duplicates=True,
    )

    # Running the genetic algorithm
    problem = RegistryProvisioningProblem()
    res = minimize(problem, algorithm, termination=("n_gen", n_gen), seed=seed, verbose=VERBOSE)#, output=MyOutput())

    # Write pickle file
    file_name = f"logs/nsgaii;ngen={n_gen};pop_size={pop_size};cross={cross_prob};mut={mutation_prob}.pkl"
    with open(file_name, "wb") as f:
        pickle.dump(res, f)

    # Parsing the genetic algorithm output
    solutions = []
    for i in range(len(res.X)):
        solution = {
            "mapping": res.X[i].tolist(),
            "mean_latency": res.F[i][0],
            "mean_provisioning_time": res.F[i][1],
            "overloaded_servers": res.CV[i][0].tolist(),
        }
        solutions.append(solution)

    best_solution = sorted(
        solutions, key=lambda solution: (solution["mean_latency"] + solution["mean_provisioning_time"])
    )[0]["mapping"]

    return best_solution


def apply_solution(solution: list):
    """Applies the registry provisioning scheme suggested by the chromosome.

    Args:
        solution (list): Registry provisioning scheme.
    """
    solution = "".join([str(int(item)) for item in solution])
    results_str = os.popen(
        f"poetry run python -m simulation -s 1 -a custom -d datasets/p2p_zero.json -n 3600 -rm {solution}"
    ).read()

    results = json.loads(results_str.replace("'", '"'))
    return results


def evaluate_solution(results):
    """Evaluates the registry provisioning scheme suggested by the chromosome.

    Returns:
        dict: Output of the evaluation function containing the fitness scores of the solution and its penalties.
    """
    mean_latency = results["mean_latency"]
    mean_provisioning_time = results["mean_provisioning_time"]

    overloaded_servers = results["overloaded_servers"]
    
    objectives = (mean_latency, mean_provisioning_time)
    penalties = (overloaded_servers)

    return (objectives, penalties)


if __name__ == "__main__":
    seed = 1
    random.seed(seed)

    best_solution = genetic_algorithm(
        n_gen=20,
        pop_size=240,
        cross_prob=0.8,
        mutation_prob=0.1,
        seed=seed,
    )

    print("=== BEST SOLUTION: ", best_solution)