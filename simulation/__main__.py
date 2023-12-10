from .utils import *
from .custom_methods import *
from .wrapper import algorithm_wrapper

import random
import argparse
import edge_sim_py


def parse_arguments(parser: argparse.ArgumentParser):
    # General arguments
    parser.add_argument("--seed", "-s", help="Seed value for EdgeSimPy", default=1, type=int)
    parser.add_argument("--algorithm", "-a", help="Resource management algorithm", type=str, required=True)
    parser.add_argument("--dataset", "-d", help="Dataset file to run the simulation", type=str, required=True)
    parser.add_argument("--number-of-steps", "-n", help="Number of steps to run the simulation", default=10, type=int)

    return parser.parse_args()


def main(seed: int, algorithm: str, dataset: str, number_of_steps: int):
    # Setting a seed value to enable reproducibility
    random.seed(seed)

    # Creating a Simulator object
    simulator = edge_sim_py.Simulator(
        dump_interval=4000,
        logs_directory=f"logs/algorithm={algorithm};dataset={dataset.split('/')[-1].split('.')[0]};seed={seed}",
        resource_management_algorithm=algorithm_wrapper,
        resource_management_algorithm_parameters={"algorithm": algorithm},
        stopping_criterion=lambda model: model.schedule.steps == number_of_steps,
        tick_duration=1,
        tick_unit="seconds",
        user_defined_functions=[pathway]
    )

    # Loading custom EdgeSimPy components and methods
    edge_sim_py.ContainerRegistry.collect = container_registry_collect
    edge_sim_py.EdgeServer.collect = edge_server_collect
    edge_sim_py.EdgeServer.can_host_container_registry = edge_server_can_host_container_registry
    edge_sim_py.NetworkFlow.collect = network_flow_collect
    edge_sim_py.NetworkLink.collect = network_link_collect
    edge_sim_py.Service.step = service_step
    edge_sim_py.Service.collect = service_collect
    edge_sim_py.Service.provision = service_provision
    edge_sim_py.User.collect = user_collect
    edge_sim_py.User.set_communication_path = user_set_communication_path
    edge_sim_py.Topology.collect = topology_collect

    # Loading conditional custom EdgeSimPy components and methods
    if "p2p" in dataset:
        edge_sim_py.EdgeServer.step = edge_server_step_with_distributed_pulling

    # Loading the dataset
    simulator.initialize(input_file=dataset)

    # Executing the simulation
    simulator.run_model()

if __name__ == "__main__":
    # Parsing command line arguments
    arguments = parse_arguments(argparse.ArgumentParser())
    
    # Executing main function
    main(
        seed=arguments.seed,
        algorithm=arguments.algorithm,
        dataset=arguments.dataset,
        number_of_steps=arguments.number_of_steps,
    )
