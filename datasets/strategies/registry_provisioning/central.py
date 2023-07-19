"""Contains the strategy for provisioning a centralized container registry."""

import networkx as nx
import edge_sim_py


def central_registry_provisioning(container_registries: list = []):
    # Get the topology
    topology = edge_sim_py.Topology.first()
    
    # Sort the nodes by closeness centrality
    # https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.closeness_centrality.html#networkx.algorithms.centrality.closeness_centrality
    closeness_centrality = nx.closeness_centrality(topology)
    sorted_closeness_centrality = sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)
    
    # Iterate over the sorted nodes by centrality and choose the most adequate one that is an edge server
    for node in sorted_closeness_centrality:
        edge_server = edge_sim_py.EdgeServer.find_by(
            attribute_name="coordinates",
            attribute_value=node[0].coordinates
        )
        if edge_server is not None:
            container_registry = edge_sim_py.provision_container_registry(
                container_registry_specification=container_registries[0],
                server=edge_server
            )
            container_registry.p2p_registry = False
            break
