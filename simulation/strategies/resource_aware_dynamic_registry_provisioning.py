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


def has_reached_replication_target(replication_data: dict, replication_target: int) -> bool:
    """Checks if the replication target has been reached.

    Args:
        replication_data (dict): Dictionary with the replication data.
        replication_target (int): Replication target.

    Returns:
        bool: True if the replication target has been reached, False otherwise.
    """
    return all([count >= replication_target for digest, count in replication_data.items()])


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

    # Skip strategy execution if it is not the time to execute it
    if current_step % execution_frequency != 0:
        return

    # Collecting edge servers that are candidates to host a container registry
    candidate_servers = get_candidate_servers(edge_servers, template_registry)

    # Calculating scores for each candidate server
    scores_metadata = calculate_scores(candidate_servers)

    # Provisioning and deprovisioning container registries
    unique_image_digests = list(set([image.digest for image in edge_sim_py.ContainerImage.all()]))
    replication_data = {image_digest: 0 for image_digest in unique_image_digests}
    while not has_reached_replication_target(replication_data, image_replicas) and scores_metadata != []:
        # Gathering variables
        metadata = scores_metadata.pop(0)
        server = metadata["edge_server"]
        server_has_container_registry = server.container_registries != []
        image_digests = [image.digest for image in server.container_images]
        images_on_server_with_replication_target_accomplished = len(
            [digest for digest in image_digests if replication_data[digest] >= image_replicas]
        )
        images_on_server_with_replication_target_unaccomplished = len(
            [digest for digest in image_digests if replication_data[digest] < image_replicas]
        )

        # Checking if at least half of the images in the server storage have not reached the replication target yet
        if images_on_server_with_replication_target_unaccomplished >= images_on_server_with_replication_target_accomplished:
            # If the server was selected and it does not have a container registry, provision it
            if not server_has_container_registry:
                container_registry = edge_sim_py.ContainerRegistry.provision(
                    target_server=server,
                    registry_cpu_demand=template_registry.cpu_demand,
                    registry_memory_demand=template_registry.memory_demand,
                )
                container_registry.p2p_registry = True

            # Updating replication data
            for image in server.container_images:
                replication_data[image.digest] += 1

            continue
        
        # If the server was not selected and it has a container registry, deprovision it
        if server_has_container_registry:
            container_registry = server.container_registries[0]
            container_registry.deprovision()

    # Deprovisioning remaining container registries
    for metadata in scores_metadata:
        server = metadata["edge_server"]
        server_has_container_registry = server.container_registries != []

        if server_has_container_registry:
            container_registry = server.container_registries[0]
            container_registry.deprovision()