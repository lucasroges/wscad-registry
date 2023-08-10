from ..utils import find_shortest_path

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


def calculate_scores(candidate_servers: list):
    """Calculates the score of each candidate server.
    
    Args:
        candidate_servers (list): List of edge servers that are candidates to host a container registry.

    Returns:
        dict: Dictionary with the score of each candidate server.    
    """
    # Gathering variables
    target_applications = edge_sim_py.Application.all()

    # For each edge server, calculate the layer matching score and number of possible recipients
    scores_metadata = []
    for candidate_server in candidate_servers:
        metadata = {
            "edge_server": candidate_server,
            "layer_matching_score": 0,
            "possible_recipients": 0
        }

        # Calculating layer matching score and number of possible recipients considering each application
        for application in target_applications:
            application_user = application.users[0]

            layer_matching_score = (
                get_layer_matching(application, candidate_server)
                if is_edge_server_closer_than_central_registry(candidate_server, application_user)
                else 0
            )
            metadata["layer_matching_score"] += layer_matching_score
            metadata["possible_recipients"] += 1 if layer_matching_score > 0 else 0

        scores_metadata.append(metadata)

    # Calculating the average number of possible recipients
    average_possible_recipients = int(sum([metadata["possible_recipients"] for metadata in scores_metadata]) / len(scores_metadata))

    # Filtering out edge servers that have recipients below the average
    scores_metadata = [metadata for metadata in scores_metadata if metadata["possible_recipients"] > average_possible_recipients]

    return scores_metadata


def p2p_enhanced_registry_provisioning(parameters: dict):
    """P2P-based enhanced registry provisioning strategy.
    This strategy dynamically (de)provisions container registries in edge servers based on the demand for container images. 

    Args:
        parameters (dict): Simulation parameters
    """
    # Gathering variables
    edge_servers = edge_sim_py.EdgeServer.all()
    current_step = edge_sim_py.Topology.first().model.schedule.steps
    template_registry = edge_sim_py.ContainerRegistry.first()

    # Gathering input parameters
    execution_frequency = 60 # seconds

    # Skip strategy execution if it is not the time to execute it
    if current_step % execution_frequency != 0:
        return
    
    # Collecting edge servers that are candidates to host a container registry
    candidate_servers = get_candidate_servers(edge_servers, template_registry)

    # Calculating scores for each candidate server
    scores_metadata = calculate_scores(candidate_servers)

    # Calculating the average layer matching score and number of possible recipients
    average_layer_matching_score = sum([metadata["layer_matching_score"] for metadata in scores_metadata]) / len(scores_metadata)

    # Filtering qualified edge servers (i.e., servers with metrics above the average)
    qualified_servers = [
        metadata["edge_server"] for metadata in scores_metadata
        if metadata["layer_matching_score"] > average_layer_matching_score
    ]

    # Provisioning and deprovisioning container registries
    for server in candidate_servers:
        is_qualified_server = server in qualified_servers
        has_container_registry = server.container_registries != []
        
        if is_qualified_server and not has_container_registry:
            container_registry = edge_sim_py.ContainerRegistry.provision(
                target_server=server,
                registry_cpu_demand=template_registry.cpu_demand,
                registry_memory_demand=template_registry.memory_demand,
            )
            container_registry.p2p_registry = True

        elif not is_qualified_server and has_container_registry:
            container_registry = server.container_registries[0]

            container_registry.deprovision()

    return