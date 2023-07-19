"""Contains a delay based strategy to initially provision services."""

import edge_sim_py
import networkx as nx


def get_candidate_hosts(entity_base_station):
    # Gathering the network topology object as we will need it later in the method
    topology = edge_sim_py.Topology.first()

    # Calculating the shortest path between the user's base station and all edge servers
    edge_servers = []
    for edge_server in edge_sim_py.EdgeServer.all():
        shortest_path = nx.shortest_path(topology, source=entity_base_station.network_switch, target=edge_server.network_switch)
        path_delay = topology.calculate_path_delay(path=shortest_path)

        edge_servers.append({"server": edge_server, "path": shortest_path, "delay": path_delay})

    # Sorting edge servers by the delay of the shortest path between their base station and the user's base station
    edge_servers = [dict_item["server"] for dict_item in sorted(edge_servers, key=lambda e: (e["delay"]))]

    return edge_servers


def delay_based_service_provisioning():
    # Iterating over all users, applications and services
    for user in edge_sim_py.User.all():
        for application in user.applications:
            for service in application.services:
                # Skipping services that are already hosted
                if service.server != None:
                    continue

                # Getting the list of edge servers sorted by the distance between their base stations and the user or previous service base station
                entity = user if service == application.services[0] else application.services[application.services.index(service) - 1].server
                edge_servers = get_candidate_hosts(entity_base_station=entity.base_station)

                # Finding the closest edge server that has resources to host the service
                for edge_server in edge_servers:
                    if edge_server.has_capacity_to_host(service=service):
                        # Updating the host's resource usage
                        edge_server.cpu_demand += service.cpu_demand
                        edge_server.memory_demand += service.memory_demand

                        # Creating relationship between the host and the registry
                        service.server = edge_server
                        edge_server.services.append(service)

                        for layer_metadata in edge_server._get_uncached_layers(service=service):
                            layer = edge_sim_py.ContainerLayer(
                                digest=layer_metadata.digest,
                                size=layer_metadata.size,
                                instruction=layer_metadata.instruction,
                            )

                            # Updating host's resource usage based on the layer size
                            edge_server.disk_demand += layer.size

                            # Creating relationship between the host and the layer
                            layer.server = edge_server
                            edge_server.container_layers.append(layer)

                        # Finding image provisioned on the edge server to get metadata
                        template_image = edge_sim_py.ContainerImage.find_by(attribute_name="digest", attribute_value=service.image_digest)
                        if template_image is None:
                            raise Exception(f"Could not find any container image with digest: {service.image_digest}")

                        # Creating the new image on the target host
                        image = edge_sim_py.ContainerImage(
                            name=template_image.name,
                            digest=template_image.digest,
                            tag=template_image.tag,
                            architecture=template_image.architecture,
                        )
                        image.layers_digests = template_image.layers_digests

                        # Connecting the new image to the target host
                        image.server = edge_server
                        edge_server.container_images.append(image)

                        break

            user.set_communication_path(app=application)