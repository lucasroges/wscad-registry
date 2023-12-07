from .utils import *
from .strategies import *

import sys
import json
import random
import argparse
import edge_sim_py


def parse_arguments(parser: argparse.ArgumentParser):
    # General arguments
    parser.add_argument("--seed", "-s", help="Seed value for reproducibility", type=int, default=1)
    parser.add_argument("--input", "-i", help="Input file to build dataset", required=True)
    parser.add_argument("--output", "-o", help="Output file to save dataset", required=True)
    parser.add_argument("--registry-provisioning", "-rp", help="Registry provisioning strategy", choices=["central", "community", "p2p"], default="central")

    # Specific arguments
    parser.add_argument("--communities", "-c", help="Number of communities", required=("community_registry" in sys.argv), type=int)

    return parser.parse_args()


def main(seed: int, input_filename: str, output_filename: str, registry_provisioning: str, communities: int):
    # Loading custom EdgeSimPy components and methods
    edge_sim_py.provision_container_registry = provision_container_registry

    # Loading base scenario     
    base_scenario = json.load(open(input_filename, "r"))

    # Setting a seed value to enable reproducibility
    random.seed(seed)

    # Creating map as hexagonal grid
    map_coordinates = create_map_coordinates(
        map_dimensions=base_scenario["map"]["dimensions"]
    )

    # Creating base stations for providing wireless connectivity to users and network switches for wired connectivity
    for cell_coordinates in map_coordinates:
        base_station = create_base_station(
            coordinates=cell_coordinates,
            wireless_delay=base_scenario["network"]["wireless_delay"]
        )
        add_connectivity_to_base_station(base_station)

    # Creating network topology
    network_topology = create_network_topology(
        **base_scenario["network"]["topology"]
    )

    # Getting edge servers coordinates using Kmeans algorithm to spread them evenly across the network
    edge_server_coordinates = get_edge_server_coordinates(
        base_scenario["edge_servers"]
    )

    # Creating edge servers
    for edge_server_spec in base_scenario["edge_servers"]["specifications"]:
        for _ in range(edge_server_spec["number_of_objects"]):
            edge_server = create_edge_server(
                specification=edge_server_spec, 
                coordinates=edge_server_coordinates.pop(0)
            )
            connect_edge_server_to_base_station(edge_server)

    # Converting layer size to megabytes and filtering out layers with size lower than 1MB
    container_images = format_container_images(base_scenario["container_images"])

    # Update container registry specification
    base_scenario["container_registries"]["specifications"]["number_of_objects"] = 1 if registry_provisioning != "community" else communities

    # Create container registries specification
    container_registries = edge_sim_py.create_container_registries(
        container_registry_specifications=[base_scenario["container_registries"]["specifications"]],
        container_image_specifications=container_images
    )

    # Saving seed state
    random_state = random.getstate()

    # Creating container registries according to the given strategy (p2p registries are allocated later, but a central registry is provisioned anyway)
    if registry_provisioning != "p2p":
        eval(f"{registry_provisioning}_registry_provisioning")(container_registries)
    else:
        central_registry_provisioning(container_registries)

    # Restoring seed state
    random.setstate(random_state)

    # Creating uniform distribution for service demands
    service_demands = []

    if ("random_requirements" in base_scenario["applications"]):
        service_demands = uniform(
            n_items=sum([application["number_of_objects"] * len(application["services"]) for application in base_scenario["applications"]["specifications"]]),
            valid_values=base_scenario["applications"]["random_requirements"]["service_demand_values"],
            shuffle_distribution=True
        )

    # Creating users
    users = create_users(base_scenario["users"], map_coordinates)

    # Creating applications and their services
    for application_spec in base_scenario["applications"]["specifications"]:
        application_user = None
        for _ in range(application_spec["number_of_objects"]):
            application = create_application(
                application_spec
            )
            for service_specification in application_spec["services"]:
                if (len(service_demands) > 0):
                    service_demand = service_demands.pop()
                    service_specification["cpu_demand"] = service_demand["cpu_demand"]
                    service_specification["memory_demand"] = service_demand["memory_demand"]
                create_application_service(application, service_specification)
            # Connecting user to application and its services
            application_user = users.pop(0)
            connect_user_to_application(application_user, application)

    # Provisioning services
    delay_based_service_provisioning()

    # Handling p2p registry provisioning, if necessary
    if registry_provisioning == "p2p":
        p2p_registry_provisioning()

    # Exporting the scenario to a file
    edge_sim_py.ContainerRegistry._to_dict = container_registry_to_dict
    edge_sim_py.ComponentManager.export_scenario(save_to_file=True, file_name=output_filename)

    # Dataset analysi
    dataset_analysis()

    # Displaying topology
    display_topology(
        edge_sim_py.Topology.first(),
        output_filepath="datasets/topologies/",
        output_filename=output_filename,
    )


if __name__ == "__main__":
    # Parsing command line arguments
    arguments = parse_arguments(argparse.ArgumentParser())

    # Executing main function
    main(
        arguments.seed,
        arguments.input,
        arguments.output,
        arguments.registry_provisioning,
        arguments.communities,
    )
