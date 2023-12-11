from .utils import *

import edge_sim_py

def custom_registry_provisioning(parameters: dict):
    """Custom registry provisioning strategy.
    This strategy (de)provisions container registries in edge servers based on the custom definitions.

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
    candidate_servers = [server for server in edge_servers if len(server.container_registries) == 0 or server.container_registries[0].p2p_registry == True]

    # Gathering custom registry mapping
    base_index = int(current_step / 60)
    registry_mapping = parameters["registry_mapping"][base_index * len(candidate_servers) : (base_index + 1) * len(candidate_servers)]

    # Filtering qualified edge servers (i.e., servers with metrics above the average)
    qualified_servers = [server for index, server in enumerate(candidate_servers) if registry_mapping[index] == 1]

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