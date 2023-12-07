from sklearn.cluster import KMeans

import copy
import random
import edge_sim_py
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


ONE_MEGABYTE = 1024 * 1024


def create_map_coordinates(map_dimensions: dict, grid_type: callable = edge_sim_py.hexagonal_grid) -> list:
    """Creates a map of the given dimensions using the given grid type.
    
    Args:
        map_dimensions (dict): A dictionary containing the width and height of the map.
        grid_type (callable): A callable that defines how the map cells are organized.
        
    Returns:
        list: A list of coordinates for the given map dimensions and grid type.
    """
    map_coordinates = grid_type(
        x_size=map_dimensions["width"],
        y_size=map_dimensions["height"]
    )

    return map_coordinates


def create_base_station(coordinates: tuple, wireless_delay) -> edge_sim_py.BaseStation:
    """Creates a base station with the given coordinates and wireless delay.

    Args:
        coordinates (tuple): A tuple containing the x and y coordinates of the base station.
        wireless_delay (float): The wireless delay of the base station.
    
    Returns:
        edge_sim_py.BaseStation: A base station with the given coordinates and wireless delay.
    """
    base_station = edge_sim_py.BaseStation()

    base_station.wireless_delay = wireless_delay
    base_station.coordinates = coordinates

    return base_station


def add_connectivity_to_base_station(base_station: edge_sim_py.BaseStation):
    """Adds connectivity to the given base station by connecting it to a network switch.

    Args:
        base_station (edge_sim_py.BaseStation): The base station to which connectivity will be added.
    """
    network_switch = edge_sim_py.sample_switch()
    base_station._connect_to_network_switch(network_switch=network_switch)


def create_network_topology(algorithm: str, link_specifications: list, **kwargs: dict):
    """Creates a network topology using the given algorithm and link specifications.

    Args:
        algorithm (str): The algorithm to be used for creating the network topology.
        link_specifications (list): A list of link specifications.
        **kwargs (dict): Additional arguments for the given algorithm.

    Returns:
        edge_sim_py.NetworkTopology: A network topology created using the given algorithm and link specifications.
    """
    if algorithm == "barabasi_albert" or algorithm == "partially_connected_hexagonal_mesh":
        return eval(f"edge_sim_py.{algorithm}")(
            network_nodes=edge_sim_py.NetworkSwitch.all(),
            link_specifications=link_specifications
        )
    
    raise ValueError(f"Algorithm {algorithm} not supported.")
        
    


def get_edge_server_coordinates(edge_servers: dict):
    """Gets the coordinates of the edge servers according to the given specifications.

    Args:
        edge_servers_specifications (dict): A dictionary containing the specifications of the edge servers.

    Returns:
        list: A list of coordinates for the edge servers.
    """
    total_number_of_edge_servers = sum([spec["number_of_objects"] for spec in edge_servers["specifications"]])

    edge_server_coordinates = []
    
    if "coordinates" in edge_servers:
        edge_server_coordinates = [(coordinates[0], coordinates[1]) for coordinates in edge_servers["coordinates"]]
    else:
        kmeans = KMeans(
            init="k-means++", 
            n_init=100, 
            n_clusters=total_number_of_edge_servers, 
            random_state=0, 
            max_iter=1000
        ).fit(
            [switch.coordinates for switch in edge_sim_py.NetworkSwitch.all()]
        )

        
        for centroid in list(kmeans.cluster_centers_):
            node_closest_to_centroid = sorted(
                edge_sim_py.NetworkSwitch.all(), key=lambda switch: np.linalg.norm(np.array(switch.coordinates) - np.array([centroid[0], centroid[1]]))
            )[0]
            edge_server_coordinates.append(node_closest_to_centroid.coordinates)
    
    return edge_server_coordinates


def create_edge_server(specification: dict, coordinates: tuple):
    """Creates an edge server with the given model name and coordinates.

    Args:
        model_name (str): The name of the model to be used for creating the edge server.
        coordinates (tuple): A tuple containing the x and y coordinates of the edge server.

    Returns:
        edge_sim_py.EdgeServer: An edge server with the given model name and coordinates.
    """
    edge_server = (
        eval(f"edge_sim_py.{specification['name']}")()
        if specification["is_implemented"]
        else edge_sim_py.EdgeServer(model_name=specification["name"], cpu=specification["cpu"], memory=specification["memory"],  disk=specification["disk"])
    )
    edge_server.coordinates = coordinates

    return edge_server

    
def connect_edge_server_to_base_station(edge_server: edge_sim_py.EdgeServer):
    """Connects the given edge server to a base station.

    Args:
        edge_server (edge_sim_py.EdgeServer): The edge server to be connected to a base station.
    """
    base_station = edge_sim_py.BaseStation.find_by(
        attribute_name="coordinates", attribute_value=edge_server.coordinates
    )
    base_station._connect_to_edge_server(edge_server=edge_server)


def format_container_images(container_images_specifications: dict):
    container_images = []
    for specification in container_images_specifications:
        container_image = {
            "name": specification["name"],
            "tag": specification["tag"],
            "digest": specification["digest"],
            "layers": [
                {
                    "digest": layer["digest"],
                    "size": layer["size"] / (1024 * 1024),
                    "instruction": layer["instruction"]
                } for layer in specification["layers"] if layer["size"] >= ONE_MEGABYTE
            ],
            "layers_digests": [layer["digest"] for layer in specification["layers"] if layer["size"] >= ONE_MEGABYTE]
        }
        container_images.append(container_image)

    return container_images
    

def uniform(n_items: int, valid_values: list, shuffle_distribution: bool = True) -> list:
    """Creates a list of size "n_items" with values from "valid_values" according to the uniform distribution.
    By default, the method shuffles the created list to avoid unbalanced spread of the distribution.

    Args:
        n_items (int): Number of items that will be created.
        valid_values (list): List of valid values for the list of values.
        shuffle_distribution (bool, optional): Defines whether the distribution is shuffled or not. Defaults to True.

    Raises:
        Exception: Invalid "valid_values" argument.

    Returns:
        uniform_distribution (list): List of values arranged according to the uniform distribution.
    """
    if not isinstance(valid_values, list) or isinstance(valid_values, list) and len(valid_values) == 0:
        raise Exception("You must inform a list of valid values within the 'valid_values' attribute.")

    # Number of occurrences that will be created of each item in the "valid_values" list
    distribution = [int(n_items / len(valid_values)) for _ in range(0, len(valid_values))]

    # List with size "n_items" that will be populated with "valid_values" according to the uniform distribution
    uniform_distribution = []

    for i, value in enumerate(valid_values):
        for _ in range(0, int(distribution[i])):
            uniform_distribution.append(value)

    # Computing leftover randomly to avoid disturbing the distribution
    leftover = n_items % len(valid_values)
    for i in range(leftover):
        random_valid_value = random.choice(valid_values)
        uniform_distribution.append(random_valid_value)

    # Shuffling distribution values in case 'shuffle_distribution' parameter is True
    if shuffle_distribution:
        random.shuffle(uniform_distribution)

    return uniform_distribution


def create_users(users_metadata: dict, map_coordinates: list):
    users = []
    for user_metadata in users_metadata:
        for _ in range(user_metadata["number_of_objects"]):
            user = edge_sim_py.User()
            user._set_initial_position(
                coordinates=random_user_placement(map_coordinates)
                if "initial_position" not in user_metadata
                else (user_metadata["initial_position"][0], user_metadata["initial_position"][1])
            )
            user.mobility_model = eval(user_metadata["mobility_model"]) if "mobility_model" in user_metadata else edge_sim_py.random_mobility
            user.coordinates_trace = user_metadata["coordinates_trace"] if "coordinates_trace" in user_metadata else []
            user.mobility_model_parameters = user_metadata["mobility_model_parameters"] if "mobility_model_parameters" in user_metadata else {}
            users.append(user)

    return users


def create_application(application_spec: dict):
    """Creates an application with the given label.
    
    Args:
        application_label (str): The label of the application to be created.

    Returns:
        edge_sim_py.Application: An application with the given label.
    """
    application = edge_sim_py.Application(label=application_spec["label"])
    
    return application


def random_user_placement(map_coordinates: list):
    """Randomly places a user in the map.

    Args:
        map_coordinates (list): A list containing the coordinates of the map.

    Returns:
        coordinates (tuple): A tuple containing the x and y coordinates of the user.
    """
    coordinates = random.choice(map_coordinates)
    return coordinates


def connect_user_to_application(user: edge_sim_py.User, application: edge_sim_py.Application):
    """Connects an user to an application with a certain delay SLA and access pattern.

    Args:
        user (edge_sim_py.User): The user to be connected to the application.
        application (edge_sim_py.Application): The application to be connected to the user.
    """
    user._connect_to_application(app=application, delay_sla=float("inf"))

    for service in application.services:
        service.users.append(user)

    edge_sim_py.CircularDurationAndIntervalAccessPattern(
        user=user,
        app=application,
        start=1,
        duration_values=[float("inf")],
        interval_values=[0],
    )


def create_application_service(application: edge_sim_py.Application, service_specifications: dict):
    """Creates an application service and connects it to the given application.

    Args:
        application (edge_sim_py.Application): The application to be connected to the service.
        service_specifications (dict): A dictionary containing the specifications of the service.
        service_demands (dict): A dictionary containing the demands of the service.
        application_user (edge_sim_py.User): The user to be connected to the service.
    """
    service = edge_sim_py.Service(
        label=service_specifications["label"],
        image_digest=service_specifications["image_digest"],
        cpu_demand=service_specifications["cpu_demand"] if "cpu_demand" in service_specifications else 0,
        memory_demand=service_specifications["memory_demand"] if "memory_demand" in service_specifications else 0,
    )

    if "initial_position" in service_specifications:
        edge_server = edge_sim_py.EdgeServer.find_by(
            attribute_name="coordinates",
            attribute_value=(
                service_specifications["initial_position"][0],
                service_specifications["initial_position"][1]
            )
        )

        # Updating the host's resource usage
        edge_server.cpu_demand += service.cpu_demand
        edge_server.memory_demand += service.memory_demand

        # Creating relationship between the host and the registry
        service.server = edge_server
        edge_server.services.append(service)

        # Creating uncached container layers
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
            layers=template_image.layers_digests
        )

        # Connecting the new image to the target host
        image.server = edge_server
        edge_server.container_images.append(image)

    # Marking the service as available
    service._available = True

    application.connect_to_service(service)


def container_registry_to_dict(self) -> dict:
    """Method that overrides the way the object is formatted to JSON."

    Returns:
        dict: JSON-friendly representation of the object as a dictionary.
    """
    dictionary = {
        "attributes": {
            "id": self.id,
            "cpu_demand": self.cpu_demand,
            "memory_demand": self.memory_demand,
            "p2p_registry": self.p2p_registry,
        },
        "relationships": {
            "server": {"class": type(self.server).__name__, "id": self.server.id} if self.server else None,
        },
    }
    return dictionary


def get_application_layers(application: edge_sim_py.Application):
    services = application.services
    images = [edge_sim_py.ContainerImage.find_by(attribute_name="digest", attribute_value=service.image_digest) for service in services]
    layers = [
        [
            edge_sim_py.ContainerLayer.find_by(attribute_name="digest", attribute_value=digest)
            for digest in image.layers_digests
        ] 
        for image in images
    ]

    unique_layers = set([layer for image_layers in layers for layer in image_layers])

    return unique_layers


def get_service_size(service: edge_sim_py.Service):
    image = edge_sim_py.ContainerImage.find_by(attribute_name="digest", attribute_value=service.image_digest)
    layers = [
        edge_sim_py.ContainerLayer.find_by(attribute_name="digest", attribute_value=digest)
        for digest in image.layers_digests
    ]

    return sum([layer.size for layer in layers])


def dataset_analysis():
    users = []
    for user in edge_sim_py.User.all():
        user_metadata = {"object": user, "all_delays": []}
        for edge_server in edge_sim_py.EdgeServer.all():
            path = nx.shortest_path(
                edge_sim_py.Topology.first(), source=user.base_station.network_switch, target=edge_server.network_switch, weight="delay"
            )
            user_metadata["all_delays"].append(edge_sim_py.Topology.first().calculate_path_delay(path=path))
        user_metadata["min_delay"] = min(user_metadata["all_delays"])
        user_metadata["max_delay"] = max(user_metadata["all_delays"])
        user_metadata["avg_delay"] = sum(user_metadata["all_delays"]) / len(user_metadata["all_delays"])
        user_metadata["delays"] = {}
        for delay in sorted(list(set(user_metadata["all_delays"]))):
            user_metadata["delays"][delay] = user_metadata["all_delays"].count(delay)

        users.append(user_metadata)

    print("==== NETWORK DISTANCE (DELAY) BETWEEN USERS AND EDGE SERVERS ====")
    for user_metadata in users:
        user_attrs = {
            "object": user_metadata["object"],
            "service_chain_size": len(user_metadata["object"].applications[0].services),
            "min": user_metadata["min_delay"],
            "max": user_metadata["max_delay"],
            "avg": round(user_metadata["avg_delay"]),
            "delays": user_metadata["delays"],
        }
        print(f"{user_attrs}")

    # Calculating the infrastructure occupation and information about the services
    edge_server_cpu_capacity = 0
    edge_server_memory_capacity = 0
    edge_server_cpu_demand = 0
    edge_server_memory_demand = 0
    service_cpu_demand = 0
    service_memory_demand = 0

    for edge_server in edge_sim_py.EdgeServer.all():
        edge_server_cpu_capacity += edge_server.cpu
        edge_server_memory_capacity += edge_server.memory
        edge_server_cpu_demand += edge_server.cpu_demand
        edge_server_memory_demand += edge_server.memory_demand

    edge_server_cpu_occupation = round((edge_server_cpu_demand / edge_server_cpu_capacity) * 100, 1)
    edge_server_memory_occupation = round((edge_server_memory_demand / edge_server_memory_capacity) * 100, 1)

    for service in edge_sim_py.Service.all():
        service_cpu_demand += service.cpu_demand
        service_memory_demand += service.memory_demand

    service_cpu_occupation = round((service_cpu_demand / edge_server_cpu_capacity) * 100, 1)
    service_memory_occupation = round((service_memory_demand / edge_server_memory_capacity) * 100, 1)

    print("\n\n==== INFRASTRUCTURE OCCUPATION OVERVIEW ====")
    print(f"Edge Servers: {edge_sim_py.EdgeServer.count()}")
    print(f"\tCPU Capacity: {edge_server_cpu_capacity}")
    print(f"\tRAM Capacity: {edge_server_memory_capacity}")
    print(f"\tCPU Demand: {edge_server_cpu_demand}")
    print(f"\tRAM Demand: {edge_server_memory_demand}")
    print(f"\tCPU Occupation: {edge_server_cpu_occupation}%")
    print(f"\tRAM Occupation: {edge_server_memory_occupation}%")
    print(f"Services: {edge_sim_py.Service.count()}")
    print(f"\tCPU Demand: {service_cpu_demand}")
    print(f"\tRAM Demand: {service_memory_demand}")
    print(f"\tCPU Occupation: {service_cpu_occupation}%")
    print(f"\tRAM Occupation: {service_memory_occupation}%")

    # Calculating the layer sharing between the images
    images = edge_sim_py.ContainerImage.all()
    unique_images = set([image.digest for image in images])
    unique_images = [edge_sim_py.ContainerImage.find_by(attribute_name="digest", attribute_value=digest) for digest in unique_images]
    
    layers = edge_sim_py.ContainerLayer.all()
    unique_layers = set([layer.digest for layer in layers])
    unique_layers = [edge_sim_py.ContainerLayer.find_by(attribute_name="digest", attribute_value=digest) for digest in unique_layers]
    unique_layers_size = sum([layer.size for layer in unique_layers])

    print("\n\n==== LAYER SHARING OVERVIEW ====")
    print(f"Images: {len(images)}")
    print(f"Unique Images: {len(unique_images)}")
    print(f"Layers: {len(layers)}")
    print(f"Unique Layers: {len(unique_layers)}")
    print(f"Unique Layers Size: {unique_layers_size}")

    layers_medata = []
    for layer in unique_layers:
        layer_metadata = {"object": layer, "size": layer.size, "images_sharing": len([image for image in unique_images if layer.digest in image.layers_digests])}
        layers_medata.append(layer_metadata)

    sharing_info = {}
    for layer_metadata in layers_medata:
        if layer_metadata["images_sharing"] not in sharing_info:
            sharing_info[layer_metadata["images_sharing"]] = layer_metadata["size"]
        else:
            sharing_info[layer_metadata["images_sharing"]] += layer_metadata["size"]

    sharing_info_percentage = {}
    total_size = sum([layer_metadata["size"] for layer_metadata in layers_medata])
    for key in sharing_info:
        sharing_info_percentage[key] = round((sharing_info[key] / total_size) * 100, 1)
    
    print(f"Shared size between N images: {sharing_info}")
    print(f"Shared size between N images (%): {sharing_info_percentage}")

    print("\n\n==== IMAGE SIZE DISTRIBUTION ====")
    image_sizes = []
    for image in unique_images:
        image_name = edge_sim_py.ContainerImage.find_by(attribute_name="digest", attribute_value=image.digest).name
        if image_name == "registry":
            continue

        image_layers = [edge_sim_py.ContainerLayer.find_by(attribute_name="digest", attribute_value=digest) for digest in image.layers_digests]
        image_size = sum([layer.size for layer in image_layers])
        image_sizes.append(image_size)

    max_image_size = max(image_sizes)
    min_image_size = min(image_sizes)
    slices = 10
    slice_size = (max_image_size - min_image_size) / slices
    image_sizes_distribution = {}
    for i in range(slices):
        image_sizes_distribution[f"{round(min_image_size + (i * slice_size), 2)} - {round(min_image_size + ((i + 1) * slice_size), 2)}"] = len([size for size in image_sizes if size >= min_image_size + (i * slice_size) and (size < min_image_size + ((i + 1) * slice_size) if i < slices - 1 else size <= min_image_size + ((i + 1) * slice_size))])

    print(f"Max image size: {max_image_size}")
    print(f"Min image size: {min_image_size}")
    print(f"Image sizes distribution: {image_sizes_distribution}")


def display_topology(topology: object, output_filepath: str, output_filename: str):
    """Method that displays the topology of the network.
    Args:
        topology (object): The topology object.
        output_filename (str, optional): The name of the output file. Defaults to "topology".
    """
    positions = {}
    labels = {}
    colors = []
    sizes = []

    # Gathering the coordinates of edge servers
    edge_server_coordinates = [edge_server.coordinates for edge_server in edge_sim_py.EdgeServer.all()]

    # Gathering the coordinates of container registries
    container_registry_coordinates = [
        edge_server.coordinates
        for edge_server
        in edge_sim_py.EdgeServer.all()
        if len([registry for registry in edge_server.container_registries if registry.available]) > 0
    ]

    for node in topology.nodes():
        labels[node] = node.id
        positions[node] = node.coordinates
        if node.coordinates in container_registry_coordinates:
            colors.append("blue")
        elif node.coordinates in edge_server_coordinates:
            colors.append("red")
        else:
            colors.append("black")

    nx.draw(
        topology,
        pos=positions,
        labels=labels,
        font_size=10,
        font_weight="bold",
        font_color="whitesmoke",
        node_color=colors,
    )

    plt.savefig(f"{output_filepath}{output_filename}.png", dpi=120)
    plt.clf()

    
def provision_container_registry(container_registry_specification: dict, server: object) -> object:
    """Creates a container registry from a dictionary with technical specifications and provisions it inside a server object.

    Args:
        container_registry_specification (dict): Container registry specification.
        server (object): Server that will host the container registry.

    Returns:
        registry (object): Provisioned container registry.
    """
    # Creating registry object
    registry = edge_sim_py.ContainerRegistry(
        cpu_demand=container_registry_specification["cpu_demand"],
        memory_demand=container_registry_specification["memory_demand"],
    )

    # Creating relationship between the edge server and the registry
    registry.server = server
    server.container_registries.append(registry)

    # Updating the host's resource usage
    server.cpu_demand += registry.cpu_demand
    server.memory_demand += registry.memory_demand

    # Creating objects representing the images and their layers, now hosted by the chosen edge server
    for image_spec in container_registry_specification["images"]:
        image = edge_sim_py.ContainerImage()
        image.name = image_spec["name"] if "name" in image_spec else ""
        image.digest = image_spec["digest"] if "digest" in image_spec else ""
        image.tag = image_spec["tag"] if "tag" in image_spec else ""
        image.architecture = image_spec["architecture"] if "architecture" in image_spec else ""
        image.layers_digests = [layer["digest"] for layer in image_spec["layers"]]

        # Creating relationship between the edge server and the image
        image.server = server
        server.container_images.append(image)

    for layer_spec in container_registry_specification["layers"]:
        layer = edge_sim_py.ContainerLayer(
            digest=layer_spec["digest"] if "digest" in layer_spec else None,
            instruction=layer_spec["instruction"] if "instruction" in layer_spec else "",
            size=layer_spec["size"] if "size" in layer_spec else 0,
        )

        # Updating edge server's resource usage based on the layer size
        server.disk_demand += layer.size

        # Creating relationship between the edge server and the layer
        layer.server = server
        server.container_layers.append(layer)

    return registry
