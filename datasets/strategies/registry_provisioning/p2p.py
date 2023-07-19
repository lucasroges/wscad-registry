"""Contains the strategy for provisioning P2P-based container registries."""

import edge_sim_py


def p2p_registry_provisioning():
    # Filter edge servers with container images and that do not have a container registry
    edge_servers_with_container_images = [
        edge_server for edge_server in edge_sim_py.EdgeServer.all()
        if len(edge_server.container_images) > 0 and len(edge_server.container_registries) == 0
    ]

    # Iterate over the filtered edge servers and provision a container registry if they have capacity
    for edge_server in edge_servers_with_container_images:
        # Template registry
        template_registry = edge_sim_py.ContainerRegistry.first()

        # Check if the edge server has capacity to host the container registry
        edge_server_free_cpu = edge_server.cpu - edge_server.cpu_demand
        edge_server_free_memory = edge_server.memory - edge_server.memory_demand
        if edge_server_free_cpu < template_registry.cpu_demand or edge_server_free_memory < template_registry.memory_demand:
            continue

        # Creating specification
        container_registry_specification = {
            "cpu_demand": template_registry.cpu_demand,
            "memory_demand": template_registry.memory_demand,
            "images": [],
            "layers": [],
        }

        # Provisioning container registry
        registry = edge_sim_py.provision_container_registry(
            container_registry_specification=container_registry_specification,
            server=edge_server
        )
        registry.p2p_registry = True

        # Getting template registry image
        template_registry_image = edge_sim_py.ContainerImage.find_by(
            attribute_name="name",
            attribute_value="registry"
        )

        # Adding registry image and layers
        registry_image = edge_sim_py.ContainerImage(
            name=template_registry_image.name,
            tag=template_registry_image.tag,
            digest=template_registry_image.digest,
            layers=template_registry_image.layers_digests,
        )

        # Creating relationship between the edge server and the image
        registry_image.server = edge_server
        edge_server.container_images.append(registry_image)

        # Getting template registry layers
        for layer_digest in template_registry_image.layers_digests:
            layer_template = edge_sim_py.ContainerLayer.find_by(
                attribute_name="digest",
                attribute_value=layer_digest
            )

            layer = edge_sim_py.ContainerLayer(
                digest=layer_template.digest,
                size=layer_template.size,
                instruction=layer_template.instruction,
            )

            # Updating edge server's resource usage based on the layer size
            edge_server.disk_demand += layer.size

            # Creating relationship between the edge server and the layer
            layer.server = edge_server
            edge_server.container_layers.append(layer)
