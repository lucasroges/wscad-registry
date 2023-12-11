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
import argparse
import random
import pickle
import json
import os

VERBOSE = True
PARALLEL = True
N_THREADS = multiprocessing.cpu_count()


def parse_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--seed", "-s", help="Seed value for EdgeSimPy", default=1, type=int)
    parser.add_argument("--number-of-generations", "-ngen", help="Number of generations the genetic algorithm will run through", type=int, required=True)
    parser.add_argument("--population-size", "-pop", help="Number of chromosomes that will represent the genetic algorithm's population", type=int, required=True)
    parser.add_argument("--crossover-probability", "-cross", help="Crossover probability", type=float, required=True)
    parser.add_argument("--mutation-probability", "-mut", help="Mutation probability", type=float, required=True)
    parser.add_argument("--dataset", "-d", help="Dataset file to run the simulation", type=str, required=True)

    return parser.parse_args()


class MyOutput(Output):
    """Creates a visualization on how the genetic algorithm is evolving throughout the generations."""
    def __init__(self):
        super().__init__()
        self.mean_latency = Column("mean_latency", width=20, truncate=False)
        self.mean_prov_time = Column("mean_prov_time", width=20, truncate=False)
        self.overloaded_servers = Column("overloaded_servers", width=20, truncate=False)
        self.columns += [self.mean_latency, self.mean_prov_time, self.overloaded_servers]

    def update(self, algorithm):
        super().update(algorithm)

        # Objective values
        mean_latency = np.min(algorithm.pop.get("F")[:, 0])
        mean_prov_time = np.min(algorithm.pop.get("F")[:, 1])
        self.mean_latency.set(mean_latency)
        self.mean_prov_time.set(mean_prov_time)

        # Penalty values
        overloaded_servers = np.min(algorithm.pop.get("CV")[:, 0])
        self.overloaded_servers.set(overloaded_servers)


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
        self.seed = kwargs["seed"]
        self.dataset = kwargs["dataset"]

    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluates solutions according to the problem objectives.
        Args:
            x (list): Solution or set of solutions that solve the problem.
            out (dict): Output of the evaluation function.
        """
        if PARALLEL:
            executor = ProcessPoolExecutor(max_workers=N_THREADS)
            futures = [executor.submit(self.get_fitness_score_and_constraints, solution, self.seed, self.dataset) for solution in x]
            output = [future.result() for future in as_completed(futures)]
            executor.shutdown()

        else:
            output = [self.get_fitness_score_and_constraints(solution=solution, seed=self.seed, dataset=self.dataset) for solution in x]

        out["F"] = np.array([item[0] for item in output])
        out["G"] = np.array([item[1] for item in output])

    def get_fitness_score_and_constraints(self, solution: list, seed: int, dataset: str) -> tuple:
        """Calculates the fitness score and penalties of a solution based on the problem definition.
        Args:
            solution (list): Solution that solves the problem.
        Returns:
            tuple: Output of the evaluation function containing the fitness scores of the solution and its penalties.
        """
        # Run the simulation with the solution
        results = apply_solution(solution, seed, dataset)

        # Gather the results
        output = evaluate_solution(results)

        return output


class NoFeasibleSolution(Exception):
    """Exception raised when the genetic algorithm does not find a feasible solution."""
    pass

        
def genetic_algorithm(
    seed: int,
    n_gen: int,
    pop_size: int,
    cross_prob: int,
    mutation_prob: int,
    dataset: str,
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
    problem = RegistryProvisioningProblem(seed=seed, dataset=dataset)
    res = minimize(
        problem,
        algorithm,
        termination=("n_gen", n_gen),
        seed=seed,
        verbose=VERBOSE,
        output=MyOutput(),
        dataset=dataset,
    )

    # Write pickle file
    file_name = f"logs/nsgaii;seed={seed};ngen={n_gen};pop_size={pop_size};cross={cross_prob};mut={mutation_prob}.pkl"
    with open(file_name, "wb") as f:
        pickle.dump(res, f)

    # Check if the algorithm has a feasible solution
    if res.X is None:
        raise NoFeasibleSolution()

    # Parsing the genetic algorithm output
    solutions = [
        {
            "mapping": solution.X,
            "mean_latency": solution.F[0],
            "mean_provisioning_time": solution.F[1],
            "overloaded_servers": solution.CV[0],
        }
        for solution in res.X
    ]

    best_solution = sorted(
        solutions, key=lambda solution: (solution["mean_latency"] + solution["mean_provisioning_time"])
    )[0]["mapping"]

    return best_solution


def apply_solution(solution: list, seed: int, dataset: str):
    """Applies the registry provisioning scheme suggested by the chromosome.

    Args:
        solution (list): Registry provisioning scheme.
    """
    solution = "".join([str(int(item)) for item in solution])
    results_str = os.popen(
        f"poetry run python -m simulation -s {seed} -a custom -d {dataset} -n 3600 -rm {solution}"
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
    # Parsing command line arguments
    arguments = parse_arguments(argparse.ArgumentParser())

    try:
        best_solution = genetic_algorithm(
            seed=arguments.seed,
            n_gen=arguments.number_of_generations,
            pop_size=arguments.population_size,
            cross_prob=arguments.crossover_probability,
            mutation_prob=arguments.mutation_probability,
            dataset=arguments.dataset,
        )

        best_solution = "".join([str(int(item)) for item in best_solution])

        print("=== BEST SOLUTION: ", best_solution)
    except NoFeasibleSolution:
        print("=== NO FEASIBLE SOLUTION FOUND")