from ..utils import find_shortest_path, normalize_cpu_and_memory

import edge_sim_py

def get_candidate_servers(edge_servers: list, template_registry: edge_sim_py.ContainerRegistry):
    """Collects the edge servers that are candidates to host a container registry.

    Args:
        edge_servers (list): List of edge servers with container images.

    Returns:
        list: List of edge servers that are candidates to host a container registry.
    """
    # Filtering out edge server with base container registry
    edge_server_with_base_registry = edge_sim_py.ContainerRegistry.find_by(attribute_name="p2p_registry", attribute_value=False).server
    
    # Filtering out edge servers without enough resources
    edge_servers_without_active_registries = [
        edge_server for edge_server in edge_servers
        if edge_server.container_registries == []
    ]
    edge_servers_without_enough_resources = [
        edge_server for edge_server in edge_servers_without_active_registries
        if not edge_server.can_host_container_registry(template_registry)
    ]

    # Collecting candidate servers
    candidate_servers = [
        edge_server for edge_server in edge_servers
        if edge_server != edge_server_with_base_registry
        and edge_server not in edge_servers_without_enough_resources
    ]

    return candidate_servers


def is_edge_server_closer_than_central_registry(edge_server: edge_sim_py.EdgeServer, user: edge_sim_py.User):
    """Checks if an edge server is closer to the user than the central registry.

    Args:
        edge_server (edge_sim_py.EdgeServer): Edge server to be checked.
        user (edge_sim_py.User): User to be checked.
    
    Returns:
        bool: True if the edge server is closer to the user than the central registry, False otherwise.
    """
    # Gathering variables
    user_base_station = user.base_station
    topology = edge_sim_py.Topology.first()
    edge_server_with_base_registry = edge_sim_py.ContainerRegistry.find_by(attribute_name="p2p_registry", attribute_value=False).server

    # Getting delay between edge server and user
    user_path_to_edge_server = find_shortest_path(
        user_base_station.network_switch,
        edge_server.network_switch
    )
    user_delay_to_edge_server = topology.calculate_path_delay(user_path_to_edge_server)

    # Getting delay between central registry and user
    user_path_to_central_registry = find_shortest_path(
        user_base_station.network_switch,
        edge_server_with_base_registry.network_switch
    )
    user_delay_to_central_registry = topology.calculate_path_delay(user_path_to_central_registry)

    return user_delay_to_edge_server < user_delay_to_central_registry


def get_layer_matching(application: edge_sim_py.Application, edge_server: edge_sim_py.EdgeServer):
    """Calculates the layer matching of an edge server to the application layers.
    
    Args:
        application (edge_sim_py.Application): Application
        edge_server (edge_sim_py.EdgeServer): Edge server

    Returns:
        float: Layer matching
    """
    # Gathering variables
    services = application.services

    # Getting layer digests of the application
    application_container_images = [
        edge_sim_py.ContainerImage.find_by(
            attribute_name="digest",
            attribute_value=service.image_digest
        ) for service in services
    ]
    application_layers_digests = []
    for image in application_container_images:
        application_layers_digests.extend(image.layers_digests)
    application_unique_layers_digests = list(set(application_layers_digests))

    # Getting layer digests of the edge server
    server_layer_digests = [layer.digest for layer in edge_server.container_layers]

    # Gathering matching layers
    application_unique_layers = [{
        "digest": digest,
        "size": edge_sim_py.ContainerLayer.find_by(attribute_name="digest", attribute_value=digest).size
    } for digest in application_unique_layers_digests]
    matching_layers = [layer for layer in application_unique_layers if layer["digest"] in server_layer_digests]

    # Calculating the layer matching percentage
    layer_matching = sum([layer["size"] for layer in matching_layers]) / sum([layer["size"] for layer in application_unique_layers])

    return layer_matching


def get_edge_server_importance_for_application(edge_server: edge_sim_py.EdgeServer, application: edge_sim_py.Application):
    """Calculates the importance of an edge server for an application.

    Args:
        edge_server (edge_sim_py.EdgeServer): Edge server
        application (edge_sim_py.Application): Application

    Returns:
        float: Importance of the edge server for the application
    """
    # Gathering variables
    app_user = application.users[0]
    topology = edge_sim_py.Topology.first()

    # Getting server occupation
    server_occupation = (
        normalize_cpu_and_memory(edge_server.cpu_demand, edge_server.memory_demand) /
        normalize_cpu_and_memory(edge_server.cpu, edge_server.memory)
    )

    # Getting current delay between user and application
    current_latency = app_user.delays[str(application.id)]

    # Getting delay between edge server and user
    user_path_to_edge_server = find_shortest_path(
        app_user.base_station.network_switch,
        edge_server.network_switch
    )
    user_delay_to_edge_server = topology.calculate_path_delay(user_path_to_edge_server)
    
    # Getting difference between current delay and delay to edge server
    latency_difference = current_latency - user_delay_to_edge_server

    # Calculating importance
    importance = latency_difference * server_occupation
    
    return importance


def get_possible_demand(edge_server: edge_sim_py.EdgeServer) -> int:
    """Get the possible demand of an edge server in future time steps.

    Args:
        edge_server (edge_sim_py.EdgeServer): Edge server

    Returns:
        int: Possible demand of the edge server in future time steps
    """
    # Gathering variables
    topology = edge_sim_py.Topology.first()
    applications = edge_sim_py.Application.all()

    # Initializing variables
    possible_demand = 0

    # Gathering possible demand
    for application in applications:
        app_user = application.users[0]
        current_latency = app_user.delays[str(application.id)]

        # Getting delay between edge server and user
        user_path_to_edge_server = find_shortest_path(
            app_user.base_station.network_switch,
            edge_server.network_switch
        )
        user_delay_to_edge_server = topology.calculate_path_delay(user_path_to_edge_server) + app_user.base_station.wireless_delay

        # Getting difference between current delay and delay to edge server
        latency_difference = current_latency - user_delay_to_edge_server

        # Gathering possible demand
        if latency_difference > 0:
            app_normalized_demand = normalize_cpu_and_memory(
                sum([service.cpu_demand for service in application.services]),
                sum([service.memory_demand for service in application.services])
            )
            possible_demand += app_normalized_demand

    return possible_demand


def get_amount_of_free_resources(edge_server: edge_sim_py.EdgeServer) -> int:
    """"Get the amount of free resource of an edge server.

    Args:
        edge_server (edge_sim_py.EdgeServer): Edge server

    Returns:
        int: Amount of free resource of the edge server
    """
    # Gathering variables
    services_leaving = [service for service in edge_server.services if service.being_provisioned]
    has_container_registry = edge_server.container_registries[0] if edge_server.container_registries != [] else None

    # Calculating amount of free resource
    free_cpu = edge_server.cpu - edge_server.cpu_demand + sum([service.cpu_demand for service in services_leaving]) + (has_container_registry.cpu_demand if has_container_registry else 0)
    free_memory = edge_server.memory - edge_server.memory_demand + sum([service.memory_demand for service in services_leaving]) + (has_container_registry.memory_demand if has_container_registry else 0)

    return normalize_cpu_and_memory(free_cpu, free_memory)