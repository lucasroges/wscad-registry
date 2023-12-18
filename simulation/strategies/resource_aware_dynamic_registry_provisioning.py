from .utils import *
from ..utils import get_geometric_mean

import edge_sim_py


def calculate_scores(candidate_servers: list):
    """Calculates the score of each candidate server.
    
    Args:
        candidate_servers (list): List of edge servers that are candidates to host a container registry.

    Returns:
        dict: Dictionary with the score of each candidate server.    
    """
    # For each edge server, calculate the layer matching score and number of possible recipients
    scores_metadata = []
    for candidate_server in candidate_servers:
        metadata = {
            "edge_server": candidate_server,
            "final_score": 0,
        }

        possible_demand = get_possible_demand(candidate_server)
        free_resources = get_amount_of_free_resources(candidate_server)

        # Calculating final score
        metadata["final_score"] = possible_demand - free_resources
        
        scores_metadata.append(metadata)

    # Sorting scores metadata by final score
    scores_metadata = sorted(scores_metadata, key=lambda metadata: metadata["final_score"])

    return scores_metadata


def get_qualified_servers(scores_metadata: list, replicas: int = 2, percentage_of_replicated_images: float = 1):
    """Gets the qualified servers.

    Args:
        scores_metadata (list): List of scores metadata.

    Returns:
        list: List of qualified servers.
    """
    # Gathering variables
    unique_image_digests = list(set([image.digest for image in edge_sim_py.ContainerImage.all()]))
    unique_image_digests = {image_digest: 0 for image_digest in unique_image_digests}
    number_of_unique_images = len(unique_image_digests)
    replication_threshold = number_of_unique_images * percentage_of_replicated_images

    # Filtering qualified servers
    servers = [metadata["edge_server"] for metadata in scores_metadata]
    qualified_servers = []
    for server in servers:
        number_of_images_with_min_replicas = len([digest for digest, count in unique_image_digests.items() if count >= replicas])

        if number_of_images_with_min_replicas > replication_threshold:
            break

        for image in server.container_images:
            if image.digest in unique_image_digests:
                unique_image_digests[image.digest] += 1

        qualified_servers.append(server)

    return qualified_servers


def resource_aware_dynamic_registry_provisioning(parameters: dict):
    """Dynamic registry provisioning strategy.
    This strategy dynamically (de)provisions container registries in edge servers based on the demand for container images and the resource demand of the edge servers.

    Args:
        parameters (dict): Simulation parameters
    """
    # Gathering variables
    edge_servers = edge_sim_py.EdgeServer.all()
    current_step = edge_sim_py.Topology.first().model.schedule.steps
    template_registry = edge_sim_py.ContainerRegistry.first()

    # Gathering input parameters
    execution_frequency = 60 # seconds
    image_replicas = parameters["replicas"]
    percentage_of_replicated_images = parameters["percentage_of_replicated_images"]

    # Skip strategy execution if it is not the time to execute it
    if current_step % execution_frequency != 0:
        return

    # Collecting edge servers that are candidates to host a container registry
    candidate_servers = get_candidate_servers(edge_servers, template_registry)

    # Calculating scores for each candidate server
    scores_metadata = calculate_scores(candidate_servers)

    # Filtering qualified edge servers (i.e., servers with metrics above the average)
    qualified_servers = get_qualified_servers(scores_metadata, image_replicas, percentage_of_replicated_images)

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