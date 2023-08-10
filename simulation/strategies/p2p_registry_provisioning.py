import edge_sim_py

def p2p_registry_provisioning(parameters: dict):
    """P2P-based registry provisioning strategy.
    This strategy dynamically provisions container registries in edge servers that have container images and do not have a container registry yet.

    Args:
        parameters (dict): Simulation parameters
    """
    # Filter edge servers with container images and that do not have a container registry
    edge_servers_with_container_images = [
        edge_server for edge_server in edge_sim_py.EdgeServer.all()
        if len(edge_server.container_images) > 0 and len(edge_server.container_registries) == 0
    ]

    template_registry = edge_sim_py.ContainerRegistry.first()

    # Iterate over the filtered edge servers and provision a container registry if they have capacity
    for edge_server in edge_servers_with_container_images:
        # Skipping servers that do not have capacity to host the container registry
        edge_server_free_cpu = edge_server.cpu - edge_server.cpu_demand
        edge_server_free_memory = edge_server.memory - edge_server.memory_demand
        if edge_server_free_cpu < template_registry.cpu_demand or edge_server_free_memory < template_registry.memory_demand:
            continue

        container_registry = edge_sim_py.ContainerRegistry.provision(target_server=edge_server)
        container_registry.p2p_registry = True