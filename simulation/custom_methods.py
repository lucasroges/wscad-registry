from .utils import *

import random
import edge_sim_py
import networkx as nx


def network_flow_collect(self) -> dict:
    """Method that collects a set of metrics for the object.

    Returns:
        metrics (dict): Object metrics.
    """
    bw = list(self.bandwidth.values())
    actual_bw = min(bw) if len([bw for bw in self.bandwidth.values() if bw == None]) == 0 else None

    if self.metadata["type"] == "layer":
        object_being_transferred = f"{str(self.metadata['object'])} ({self.metadata['object'].instruction})"
    else:
        object_being_transferred = str(self.metadata["object"])

    if self.status == "finished" and self.end == self.model.schedule.steps + 1:
        metrics = {
            "Instance ID": self.id,
            "Object being Transferred": object_being_transferred,
            "Object Type": self.metadata["type"],
            "Start": self.start,
            "End": self.end,
            "Source": self.source.id if self.source else None,
            "Target": self.target.id if self.target else None,
            "Path": [node.id for node in self.path],
            "Links Bandwidth": bw,
            "Actual Bandwidth": actual_bw,
            "Status": self.status,
            "Data Size": self.metadata["size"],
        }
        return metrics

    return {}


def network_link_collect(self) -> dict:
    """Method that collects a set of metrics for the object.

    Returns:
        metrics (dict): Object metrics.
    """
    metrics = {
        "Instance ID": self.id,
        "Bandwidth Demand": sum([flow.data_to_transfer for flow in self.active_flows]),
        "Number of Active Flows": len(self.active_flows),
    }
    return metrics


def service_step(self):
    """Method that executes the events involving the object at each time step."""
    if len(self._Service__migrations) > 0 and self._Service__migrations[-1]["end"] == None:
        migration = self._Service__migrations[-1]

        # Gathering information about the service's image
        image = edge_sim_py.ContainerImage.find_by(attribute_name="digest", attribute_value=self.image_digest)

        # Gathering layers present in the target server (layers, download_queue, waiting_queue)
        layers_downloaded = [l for l in migration["target"].container_layers if l.digest in image.layers_digests]
        layers_on_download_queue = [
            flow.metadata["object"]
            for flow in migration["target"].download_queue
            if flow.metadata["object"].digest in image.layers_digests
        ]

        # Setting the migration status to "pulling_layers" once any of the service layers start being downloaded
        if migration["status"] == "waiting":
            layers_on_target_server = layers_downloaded + layers_on_download_queue

            if len(layers_on_target_server) > 0:
                migration["status"] = "pulling_layers"

        if migration["status"] == "pulling_layers" and len(image.layers_digests) == len(layers_downloaded):
            # Once all the layers that compose the service's image are pulled, the service container is deprovisioned on its
            # origin host even though it still is in there (that's why it is still on the origin's services list). This action
            # is only taken in case the current provisioning process regards a migration.
            if self.server:
                self.server.cpu_demand -= self.cpu_demand
                self.server.memory_demand -= self.memory_demand

            # Once all service layers have been pulled, creates a ContainerImage object representing
            # the service image on the target host if that host didn't already have such image
            if not any([image.digest == self.image_digest for image in migration["target"].container_images]):
                # Finding similar image provisioned on the infrastructure to get metadata from it when creating the new image
                template_image = edge_sim_py.ContainerImage.find_by(attribute_name="digest", attribute_value=self.image_digest)
                if template_image is None:
                    raise Exception(f"Could not find any container image with digest: {self.image_digest}")

                # Creating the new image on the target host
                image = edge_sim_py.ContainerImage()
                image.name = template_image.name
                image.digest = template_image.digest
                image.tag = template_image.tag
                image.architecture = template_image.architecture
                image.layers_digests = template_image.layers_digests

                # Connecting the new image to the target host
                image.server = migration["target"]
                migration["target"].container_images.append(image)

            if self.state == 0 or self.server == None:
                # Stateless Services: migration is set to finished immediately after layers are pulled
                migration["status"] = "finished"
            else:
                # Stateful Services: state must be migrated to the target host after layers are pulled
                migration["status"] = "migrating_service_state"

                # Services are unavailable during the period where their states are being migrated
                self._available = False

                # Selecting the path that will be used to transfer the service state
                path = nx.shortest_path(
                    self.model.topology,
                    source=self.server.base_station.network_switch,
                    target=migration["target"].base_station.network_switch,
                )

                # Creating network flow representing the service state that will be migrated to its target host
                flow = edge_sim_py.NetworkFlow(
                    topology=self.model.topology,
                    source=self.server,
                    target=migration["target"],
                    start=self.model.schedule.steps + 1,
                    path=path,
                    data_to_transfer=self.state,
                    metadata={"type": "service_state", "object": self, "size": self.state},
                )
                self.model.initialize_agent(agent=flow)

        # Incrementing the migration time metadata
        if migration["status"] == "waiting":
            migration["waiting_time"] += 1
        elif migration["status"] == "pulling_layers":
            migration["pulling_layers_time"] += 1
        elif migration["status"] == "migrating_service_state":
            migration["migrating_service_state_time"] += 1

        if migration["status"] == "finished":
            # Storing when the migration has finished
            migration["end"] = self.model.schedule.steps + 1

            # Updating the service's origin server metadata
            if self.server:
                self.server.services.remove(self)
                self.server.ongoing_migrations -= 1

            # Updating the service's target server metadata
            self.server = migration["target"]
            self.server.services.append(self)
            self.server.ongoing_migrations -= 1

            # Tagging the service as available once their migrations finish
            self._available = True
            self.being_provisioned = False

            # Changing the routes used to communicate the application that owns the service to its users
            app = self.application
            users = app.users
            for user in users:
                user.set_communication_path(app)


def service_collect(self) -> dict:
    """Method that collects a set of metrics for the object.

    Returns:
        metrics (dict): Object metrics.
    """
    number_of_migrations = len(self._Service__migrations)
    number_of_finished_migrations = len([migration for migration in self._Service__migrations if migration["status"] == "finished"])
    number_of_finished_migrations_without_using_cache = len(
        [migration for migration in self._Service__migrations if migration["status"] == "finished" and migration["waiting_time"] + migration["pulling_layers_time"] > 0]
    )

    total_waiting_time = sum([migration["waiting_time"] for migration in self._Service__migrations])
    total_pulling_layers_time = sum([migration["pulling_layers_time"] for migration in self._Service__migrations])

    migrations_duration = [migration["end"] - migration["start"] for migration in self._Service__migrations if migration["status"] == "finished"]
    migrations_without_cache_duration = [migration["end"] - migration["start"] for migration in self._Service__migrations if migration["status"] == "finished" and migration["waiting_time"] + migration["pulling_layers_time"] > 0]

    metrics = {
        "Instance ID": self.id,
        "Available": self._available,
        "Server": self.server.id if self.server else None,
        "Being Provisioned": self.being_provisioned,
        "Number of Migrations": number_of_migrations,
        "Number of Finished Migrations": number_of_finished_migrations,
        "Number of Finished Migrations Without Using Cache": number_of_finished_migrations_without_using_cache,
        "Total Waiting Time": total_waiting_time,
        "Total Pulling Layers Time": total_pulling_layers_time,
        "Migrations Duration": migrations_duration,
        "Migrations Without Cache Duration": migrations_without_cache_duration,
    }
    return metrics


def service_provision(self, target_server: object):
    """Starts the service's provisioning process. This process comprises both placement and migration. In the former, the
    service is not initially hosted by any server within the infrastructure. In the latter, the service is already being
    hosted by a server and we want to relocate it to another server within the infrastructure.

    The difference from EdgeSimPy's default method is that this custom method identifies the services being provisioned at the target server.

    Args:
        target_server (object): Target server.
    """
    # Gathering layers present in the target server (layers, download_queue, waiting_queue)
    layers_downloaded = [layer for layer in target_server.container_layers]
    layers_on_download_queue = [flow.metadata["object"] for flow in target_server.download_queue]
    layers_on_waiting_queue = [layer for layer in target_server.waiting_queue]

    layers_on_target_server = layers_downloaded + layers_on_download_queue + layers_on_waiting_queue

    # Gathering the list of layers that compose the service image that are not present in the target server
    image = edge_sim_py.ContainerImage.find_by(attribute_name="digest", attribute_value=self.image_digest)
    for layer_digest in image.layers_digests:
        if not any(layer.digest == layer_digest for layer in layers_on_target_server):
            # As the image only stores its layers digests, we need to get information about each of its layers
            layer_metadata = edge_sim_py.ContainerLayer.find_by(attribute_name="digest", attribute_value=layer_digest)

            # Creating a new layer object that will be pulled to the target server
            layer = edge_sim_py.ContainerLayer(
                digest=layer_metadata.digest,
                size=layer_metadata.size,
                instruction=layer_metadata.instruction,
            )
            self.model.initialize_agent(agent=layer)

            # Reserving the layer disk demand inside the target server
            target_server.disk_demand += layer.size

            # Adding the layer to the target server's waiting queue (layers it must download at some point)
            target_server.waiting_queue.append(layer)

    # Telling EdgeSimPy that this service is being provisioned
    self.being_provisioned = True

    # Telling EdgeSimPy the service's current server is now performing a migration. This action is only triggered in case
    # this method is called for performing a migration (i.e., the service is already within the infrastructure)
    if self.server:
        self.server.ongoing_migrations += 1

    # Reserving the service demand inside the target server and telling EdgeSimPy that server will receive a service
    target_server.ongoing_migrations += 1
    target_server.cpu_demand += self.cpu_demand
    target_server.memory_demand += self.memory_demand

    # Updating the service's migration status
    self._Service__migrations.append(
        {
            "status": "waiting",
            "origin": self.server,
            "target": target_server,
            "start": self.model.schedule.steps + 1,
            "end": None,
            "waiting_time": 0,
            "pulling_layers_time": 0,
            "migrating_service_state_time": 0,
        }
    )


def user_set_communication_path(self, app: object, communication_path: list = []) -> list:
    """Updates the set of links used during the communication of user and its application.

    Args:
        app (object): User application.
        communication_path (list, optional): User-specified communication path. Defaults to [].

    Returns:
        communication_path (list): Updated communication path.
    """
    topology = edge_sim_py.Topology.first()

    # Releasing links used in the past to connect the user with its application
    if app in self.communication_paths:
        path = [[edge_sim_py.NetworkSwitch.find_by_id(i) for i in p] for p in self.communication_paths[str(app.id)]]
        topology._release_communication_path(communication_path=path, app=app)

    # Defining communication path
    if len(communication_path) > 0:
        self.communication_paths[str(app.id)] = communication_path
    else:
        self.communication_paths[str(app.id)] = []

        service_hosts_base_stations = [service.server.base_station for service in app.services if service.server]
        communication_chain = [self.base_station] + service_hosts_base_stations

        # Defining a set of links to connect the items in the application's service chain
        for i in range(len(communication_chain) - 1):

            # Defining source and target nodes
            source = communication_chain[i]
            target = communication_chain[i + 1]

            # Finding and storing the best communication path between the source and target nodes
            if source == target:
                path = []
            else:
                path = find_shortest_path(source=source.network_switch, target=target.network_switch)

            # Adding the best path found to the communication path
            self.communication_paths[str(app.id)].append([network_switch.id for network_switch in path])

            # Computing the new demand of chosen links
            path = [[edge_sim_py.NetworkSwitch.find_by_id(i) for i in p] for p in self.communication_paths[str(app.id)]]
            topology._allocate_communication_path(communication_path=path, app=app)

    # Computing application's delay
    self._compute_delay(app=app, metric="latency")

    communication_path = self.communication_paths[str(app.id)]
    return communication_path


def user_collect(self) -> dict:
    """Method that collects a set of metrics for the object.

    Returns:
        metrics (dict): Object metrics.
    """
    # Computing application metrics
    application_cpu_demand = sum([service.cpu_demand for service in self.applications[0].services])
    application_memory_demand = sum([service.memory_demand for service in self.applications[0].services])

    metrics = {
        "Instance ID": self.id,
        "Coordinates": self.coordinates,
        "Base Station": f"{self.base_station} ({self.base_station.coordinates})" if self.base_station else None,
        "Delays": sum(self.delays.values()),
        "Application CPU Demand": application_cpu_demand,
        "Application Memory Demand": application_memory_demand,
        "User Type": self.type,
    }
    return metrics


# TODO: add replication metrics
def topology_collect(self) -> dict:
    """Method that collects a set of metrics for the object.

    Returns:
        metrics (dict): Object metrics.
    """

    # Declaring infrastructure metrics
    overloaded_edge_servers = 0
    overall_occupation = 0
    occupation_per_model = {}
    active_servers_per_model = {}

    # Collecting infrastructure metrics
    for edge_server in edge_sim_py.EdgeServer.all():
        # Overall Occupation
        capacity = normalize_cpu_and_memory(cpu=edge_server.cpu, memory=edge_server.memory)
        demand = normalize_cpu_and_memory(cpu=edge_server.cpu_demand, memory=edge_server.memory_demand)
        overall_occupation += demand / capacity * 100

        # Number of overloaded edge servers
        free_cpu = edge_server.cpu - edge_server.cpu_demand
        free_memory = edge_server.memory - edge_server.memory_demand
        free_disk = edge_server.disk - edge_server.disk_demand
        if free_cpu < 0 or free_memory < 0 or free_disk < 0:
            overloaded_edge_servers += 1

        # Occupation per Server Model
        if edge_server.model_name not in occupation_per_model.keys():
            occupation_per_model[edge_server.model_name] = []
        occupation_per_model[edge_server.model_name].append(demand / capacity * 100)

    # Aggregating overall metrics
    overall_occupation = overall_occupation / edge_sim_py.EdgeServer.count()

    for model_name in occupation_per_model.keys():
        active_servers_per_model[model_name] = len([item for item in occupation_per_model[model_name] if item > 0])
        occupation_per_model[model_name] = sum(occupation_per_model[model_name]) / len(occupation_per_model[model_name])            

    metrics = {
        "Overloaded Edge Servers": overloaded_edge_servers,
        "Overall Occupation": overall_occupation,
        "Occupation Per Model": occupation_per_model,
        "Active Servers Per Model": active_servers_per_model
    }

    return metrics


def edge_server_can_host_container_registry(self, registry: object) -> bool:
    """Checks if the edge server has container images and has enough free resources to host a given registry.

    Args:
        registry (object): Registry object.

    Returns:
        can_host (bool): Information of whether the edge server has capacity to host the registry or not.
    """
    # Checking if the edge server has container images
    if len(self.container_images) == 0:
        return False

    # Calculating the additional disk demand that would be incurred to the edge server
    registry_image = edge_sim_py.ContainerImage.find_by(attribute_name="name", attribute_value="registry")

    edge_server_has_registry_image = any([
        image.digest == registry_image.digest for image in self.container_images
    ])    

    additional_disk_demand = 0 if edge_server_has_registry_image else sum([
        edge_sim_py.ContainerLayer.find_by(attribute_name="digest", attribute_value=digest).size for digest in registry_image.layers_digests
    ])

    # Calculating the edge server's free resources
    free_cpu = self.cpu - self.cpu_demand
    free_memory = self.memory - self.memory_demand
    free_disk = self.disk - self.disk_demand

    # Checking if the host would have resources to host the registry and its (additional) layers
    can_host = free_cpu >= registry.cpu_demand and free_memory >= registry.memory_demand and free_disk >= additional_disk_demand
    return can_host


def container_registry_collect(self) -> dict:
    """Method that collects a set of metrics for the object.

    Returns:
        metrics (dict): Object metrics.
    """
    flows_using_the_registry = [
        flow for flow in edge_sim_py.NetworkFlow.all() if flow.status == "active" and flow.metadata["container_registry"] == self
    ]

    metrics = {
        "Available": self.available,
        "CPU Demand": self.cpu_demand,
        "RAM Demand": self.memory_demand,
        "Server": self.server.id if self.server else None,
        "Images": [image.id for image in self.server.container_images] if self.server else [],
        "Layers": [layer.id for layer in self.server.container_layers] if self.server else [],
        "P2P": self.p2p_registry,
        "Provisioning": 1 if len(flows_using_the_registry) > 0 else 0,
        "Not Provisioning": 0 if len(flows_using_the_registry) > 0 else 1,
    }
    return metrics


def edge_server_collect(self) -> dict:
    """Method that collects a set of metrics for the object.
    
    Returns:
        metrics (dict): Object metrics.
    """
    metrics = {
        "Instance ID": self.id,
        "Coordinates": self.coordinates,
        "Available": self.available,
        "CPU": self.cpu,
        "RAM": self.memory,
        "Disk": self.disk,
        "CPU Demand": self.cpu_demand,
        "RAM Demand": self.memory_demand,
        "Disk Demand": self.disk_demand,
    }
    return metrics


def edge_server_step_with_distributed_pulling(self):
    """Method that executes the events involving the object at each time step."""
    current_step = self.model.schedule.steps
    while len(self.waiting_queue) > 0 and len(self.download_queue) < self.max_concurrent_layer_downloads:
        layer = self.waiting_queue.pop(0)

        # Gathering the list of registries that have the layer
        registries_with_layer = []
        for registry in [reg for reg in edge_sim_py.ContainerRegistry.all() if reg.available]:
            # Checking if the registry is hosted on a valid host in the infrastructure and if it has the layer we need to pull
            if registry.server and any(layer.digest == l.digest for l in registry.server.container_layers):
                # Selecting a network path to be used to pull the layer from the registry
                path = find_shortest_path(
                    source=registry.server.base_station.network_switch,
                    target=self.base_station.network_switch
                )

                # Calculating how many layers the registry is provisioning
                flows_using_the_registry = len([
                    flow for flow in edge_sim_py.NetworkFlow.all() if flow.status == "active" and flow.metadata["container_registry"] == registry
                ])

                registries_with_layer.append({"object": registry, "path": path, "queue_size": flows_using_the_registry})

        # Selecting the registry from which the layer will be pulled to the (target) edge server
        registries_with_layer = sorted(registries_with_layer, key=lambda r: (r["queue_size"], len(r["path"])))
        registry = registries_with_layer[0]["object"]
        path = registries_with_layer[0]["path"]

        # Creating the flow object
        flow = edge_sim_py.NetworkFlow(
            topology=self.model.topology,
            source=registry.server,
            target=self,
            start=self.model.schedule.steps + 1,
            path=path,
            data_to_transfer=layer.size,
            metadata={"type": "layer", "object": layer, "container_registry": registry},
        )
        self.model.initialize_agent(agent=flow)

        # Adding the created flow to the edge server's download queue
        self.download_queue.append(flow)


def pathway(user: object):
    """Creates a mobility path for an user based on the Pathway mobility model.

    Args:
        user (object): User whose mobility will be defined.
    """
    # Defining the mobility model parameters based on the user's 'mobility_model_parameters' attribute
    if hasattr(user, "mobility_model_parameters"):
        parameters = user.mobility_model_parameters
    else:
        parameters = {}

    # Number of "mobility routines" added each time the method is called. Defaults to 1.
    n_paths = parameters["n_paths"] if "n_paths" in parameters else 1

    # Gathering the BaseStation located in the current client's location
    current_node = edge_sim_py.BaseStation.find_by(attribute_name="coordinates", attribute_value=user.coordinates)

    # Defining the user's mobility path
    mobility_path = []

    for i in range(n_paths):
        # Defining a target location and gathering the BaseStation located in that location
        target_node = random.choice([bs for bs in edge_sim_py.BaseStation.all() if bs != current_node])

        # Calculating the shortest mobility path according to the Pathway mobility model
        path = nx.shortest_path(G=user.model.topology, source=current_node.network_switch, target=target_node.network_switch)
        mobility_path.extend([network_switch.base_station for network_switch in path])

        if i < n_paths - 1:
            current_node = mobility_path.pop(-1)

        # Removing repeated entries
        user_base_station = edge_sim_py.BaseStation.find_by(attribute_name="coordinates", attribute_value=user.coordinates)
        if user_base_station == mobility_path[0] and user.model.schedule.steps > 0:
            mobility_path.pop(0)

    # We assume that users do not necessarily move from one step to another, as one step may represent a very small time interval
    # (e.g., 1 millisecond). Therefore, each position on the mobility path is repeated N times, so that user takes a predefined
    # amount of time steps to move from one position to another. By default, users take at least 1 minute to move across positions
    # in the map. This parameter can be changed by passing a "seconds_to_move" key to the "parameters" parameter.
    if "seconds_to_move" in parameters and type(parameters["seconds_to_move"]) == int and parameters["seconds_to_move"] < 1:
        raise Exception("The 'seconds_to_move' key passed inside the mobility model's 'parameters' attribute must be > 1.")

    seconds_to_move = parameters["seconds_to_move"] if "seconds_to_move" in parameters else 60
    seconds_to_move = max([1, int(seconds_to_move / user.model.tick_duration)])

    mobility_path = [item for item in mobility_path for i in range(seconds_to_move)]

    # Adding the path that connects the current to the target location to the client's mobility trace
    user.coordinates_trace.extend([bs.coordinates for bs in mobility_path])

