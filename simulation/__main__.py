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

    # Specific arguments
    parser.add_argument("--replicas", "-r", help="Number of replicas", default=1, type=int)
    parser.add_argument("--percentage-of-replicated-images", "-p", help="Percentage of replicated images", default=1, type=float)
    parser.add_argument("--registry-mapping", "-rm", help="Registry mapping", default=None, type=str)

    return parser.parse_args()


def main(
    seed: int,
    algorithm: str,
    dataset: str,
    number_of_steps: int,
    replicas: int,
    percentage_of_replicated_images: float,
    registry_mapping: str
):
    # Setting a seed value to enable reproducibility
    random.seed(seed)

    # Setting algorithm parameters
    algorithm_parameters = {
        "algorithm": algorithm,
        "replicas": replicas,
        "percentage_of_replicated_images": percentage_of_replicated_images,
        "registry_mapping": list(map(int, registry_mapping)) if registry_mapping is not None else None
    }

    # Creating a Simulator object
    simulator = edge_sim_py.Simulator(
        dump_interval=4000,
        logs_directory=f"logs/algorithm={algorithm};dataset={dataset.split('/')[-1].split('.')[0]};seed={seed}" if algorithm != "resource_aware_dynamic" else f"logs/algorithm={algorithm};dataset={dataset.split('/')[-1].split('.')[0]};seed={seed};replicas={replicas};percentage={percentage_of_replicated_images}",
        resource_management_algorithm=algorithm_wrapper,
        resource_management_algorithm_parameters=algorithm_parameters,
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

    # Printing the simulation results
    results = {
        "mean_latency": get_mean_latency(simulator=simulator),
        "mean_provisioning_time": get_mean_provisioning_time(simulator=simulator),
        "overloaded_servers": get_overloaded_edge_servers(simulator=simulator)
    }
    print(results)

if __name__ == "__main__":
    # Parsing command line arguments
    arguments = parse_arguments(argparse.ArgumentParser())
    
    # Executing main function
    main(
        seed=arguments.seed,
        algorithm=arguments.algorithm,
        dataset=arguments.dataset,
        number_of_steps=arguments.number_of_steps,
        replicas=arguments.replicas,
        percentage_of_replicated_images=arguments.percentage_of_replicated_images,
        registry_mapping=arguments.registry_mapping
    )
