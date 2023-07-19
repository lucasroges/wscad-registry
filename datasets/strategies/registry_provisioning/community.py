"""Contains the strategy for provisioning container registries in communities, based on Knob et al. [1].

[1] Knob, L. A. D., Faticanti, F., Ferreto, T., & Siracusa, D. (2021, October). Community-based placement of registries to speed up application deployment on Edge Computing. In 2021 IEEE International Conference on Cloud Engineering (IC2E) (pp. 147-153). IEEE.
"""

import edge_sim_py
import networkx as nx


def community_registry_provisioning(container_registries: list = []):
    # Get the topology
    topology = edge_sim_py.Topology.first()

    # Partition graph into communities using the Fluid Communities algorithm
    communities = nx.community.asyn_fluidc(topology, len(container_registries))
    
    # For each community, find the most central node using the eigenvector centrality algorithm
    for community_index, community in enumerate(communities):
        community_topology = topology.subgraph(community)
        eigenvector_centrality = nx.eigenvector_centrality(community_topology, max_iter=1000)
        sorted_eigenvector_centrality = sorted(eigenvector_centrality.items(), key=lambda x: x[1], reverse=True)

        # Iterate over the sorted nodes by centrality and filter in the edge servers
        for node in sorted_eigenvector_centrality:
            edge_server = edge_sim_py.EdgeServer.find_by(
                attribute_name="coordinates",
                attribute_value=node[0].coordinates
            )
            if edge_server is not None:
                container_registry = edge_sim_py.provision_container_registry(
                    container_registry_specification=container_registries[community_index],
                    server=edge_server
                )
                container_registry.p2p_registry = False
                break
