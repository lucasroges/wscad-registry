from . utils import *

import edge_sim_py


def calculate_scores(candidate_servers: list):
    """Calculates the score of each candidate server.
    
    Args:
        candidate_servers (list): List of edge servers that are candidates to host a container registry.

    Returns:
        dict: Dictionary with the score of each candidate server.    
    """
    # Gathering variables
    target_applications = edge_sim_py.Application.all()
    topology = edge_sim_py.Topology.first()

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
            user_path_to_edge_server = find_shortest_path(
                application_user.base_station.network_switch,
                candidate_server.network_switch
            )
            user_delay_to_edge_server = topology.calculate_path_delay(user_path_to_edge_server)

            layer_matching_score = (
                get_layer_matching(application, candidate_server) / (user_delay_to_edge_server + 1)
            )
            metadata["layer_matching_score"] += layer_matching_score
            metadata["possible_recipients"] += 1 if layer_matching_score > 0 else 0

        scores_metadata.append(metadata)

    return scores_metadata


def get_qualified_servers(scores_metadata: list):
    """Filters out edge servers that have metrics below the average.

    Args:
        scores_metadata (list): List of dictionaries with the score of each candidate server.

    Returns:
        list: List of edge servers that are qualified to host a container registry.
    """
    # Calculating the average number of possible recipients
    average_possible_recipients = int(sum([metadata["possible_recipients"] for metadata in scores_metadata]) / len(scores_metadata))

    # Filtering out edge servers that have recipients below the average
    scores_metadata = [metadata for metadata in scores_metadata if metadata["possible_recipients"] > average_possible_recipients]

    # Calculating the average layer matching score
    average_layer_matching_score = sum([metadata["layer_matching_score"] for metadata in scores_metadata]) / len(scores_metadata)

    # Filtering out edge servers that have metrics below the average
    qualified_servers = [
        metadata["edge_server"] for metadata in scores_metadata
        if metadata["layer_matching_score"] > average_layer_matching_score
    ]

    return qualified_servers


def dynamic_enhanced_registry_provisioning(parameters: dict):
    """Dynamic registry provisioning strategy.
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

    # Filtering qualified edge servers (i.e., servers with metrics above the average)
    qualified_servers = get_qualified_servers(scores_metadata)

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