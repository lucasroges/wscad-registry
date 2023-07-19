from .strategies import p2p_registry_provisioning

import random
import edge_sim_py
import networkx as nx


def get_candidate_hosts(entity_base_station: edge_sim_py.BaseStation) -> list:
    """Computes the list of edge servers sorted by the delay-based distance between their base stations and the user or previous service base station.

    Args:
        entity_base_station (edge_sim_py.BaseStation): Base station of the user or previous service in the app chain

    Returns:
        list: List of edge servers sorted by the delay-based distance between their base stations and the user or previous service base station
    """
    topology = edge_sim_py.Topology.first()
    edge_servers = []

    # Calculating the shortest path between the user's base station and each edge server's base station, based on network delay
    for edge_server in edge_sim_py.EdgeServer.all():
        shortest_path = find_shortest_path(entity_base_station.network_switch, edge_server.network_switch)
        path_delay = topology.calculate_path_delay(path=shortest_path)
        edge_servers.append({"server": edge_server, "path": shortest_path, "delay": path_delay})

    # Sorting edge servers by the delay of the shortest path between their base stations and the user or previous service base station
    edge_servers = [dict_item["server"] for dict_item in sorted(edge_servers, key=lambda e: (e["delay"]))]

    return edge_servers


def delay_based_follow_user():
    """Simple strategy that keeps application services as close as possible to each other and to the end-user.
    However, this adapted strategy only migrates when the delay is greater than or equal to the delay SLA.
    It migrates user's services to the edge server closest to the base station used by the user or the previous service in the app chain
    (we use network delay as proximity measure, as shown in the sorting process from get_candidate_hosts function).
    """
    # Iterating over all users
    for user in edge_sim_py.User.all():

        # Selecting applications that are violating delay SLA or near it (i.e., the delay between the user and the application is greater than or equal to the delay SLA times the delay threshold)
        target_applications = [app for app in user.applications if user.delays[str(app.id)] >= user.delay_slas[str(app.id)]]

        for application in target_applications:
            for service in application.services:
                # Skipping services that are already being migrated
                if len(service._Service__migrations) > 0 and service._Service__migrations[-1]["status"] != "finished":
                    continue
                # Getting the list of edge servers sorted by the distance between their base stations and the user or previous service base station
                entity = user if service == application.services[0] else (
                    application.services[application.services.index(service) - 1]._Service__migrations[-1]["target"]
                    if (
                        len(application.services[application.services.index(service) - 1]._Service__migrations) > 0 and
                        application.services[application.services.index(service) - 1]._Service__migrations[-1]["status"] != "finished"
                    )
                    else application.services[application.services.index(service) - 1].server
                )
                
                edge_servers = get_candidate_hosts(entity_base_station=entity.base_station)

                # Finding the closest edge server that has resources to host the service
                for edge_server in edge_servers:
                    # Stops the search in case the edge server that hosts the service is already the closest to the user
                    if edge_server == service.server:
                        break
                    # Checks if the edge server has resources to host the service
                    elif edge_server.has_capacity_to_host(service):
                        service.provision(target_server=edge_server)
                        break

            # Updating the routes used by the user to communicate with his applications
            user.set_communication_path(app=application)


def algorithm_wrapper(parameters: dict):
    """Wrapper function to store random state for different datasets while the algorithm runs.

    Args:
        parameters (dict): Strategy parameters
    """
    # Saving the random state to restore it later because the code below uses random and may variate between different strategies of container registry provisioning and/or network scheduling
    random_state = random.getstate()

    # Running the custom algorithm
    try:
        eval(f"{parameters['algorithm']}_registry_provisioning")(parameters=parameters)
    except NameError:
        print(f"{parameters['algorithm']} strategy does not require a custom algorithm.")

    # Running the service reallocation algorithm
    delay_based_follow_user()

    # Restoring the random state
    random.setstate(random_state)


def least_congested_shortest_path(topology: edge_sim_py.Topology, source: edge_sim_py.NetworkSwitch, target: edge_sim_py.NetworkSwitch) -> list:
    """Finds the least congested shortest path between two network switches.

    Args:
        topology (edge_sim_py.Topology): Network topology
        source (edge_sim_py.NetworkSwitch): Source network switch
        target (edge_sim_py.NetworkSwitch): Target network switch

    Returns:
        list: Least congested shortest path between the source and target network switches    
    """
    shortest_paths = nx.all_shortest_paths(
        topology,
        source=source,
        target=target,
        weight="delay"
    )

    paths = []
    for path in shortest_paths:
        # Gathering data to calculate the maximum data to transfer
        max_data_to_transfer = 0
        for i in range(len(path) - 1):
            # Finding the network link
            network_link = (
                edge_sim_py.NetworkLink.find_by(attribute_name="nodes", attribute_value=[path[i], path[i + 1]])
                if edge_sim_py.NetworkLink.find_by(attribute_name="nodes", attribute_value=[path[i], path[i + 1]]) is not None
                else edge_sim_py.NetworkLink.find_by(attribute_name="nodes", attribute_value=[path[i + 1], path[i]])
            )
            # Calculating the maximum data to transfer
            for active_flow in network_link.active_flows:
                if active_flow.data_to_transfer > max_data_to_transfer:
                    max_data_to_transfer = active_flow.data_to_transfer

        paths.append({
            "object": source,
            "path": path,
            "max_data_to_transfer": max_data_to_transfer
        })

    return min(paths, key=lambda path: path["max_data_to_transfer"])


def find_shortest_path(origin: edge_sim_py.NetworkSwitch, target: edge_sim_py.NetworkSwitch) -> list:
    """Finds the shortest path (delay used as weight) between two network switches (origin and target).

    Args:
        origin (edge_sim_py.NetworkSwitch): Origin network switch.
        target (edge_sim_py.NetworkSwitch): Target network switch.

    Returns:
        path (list): Shortest path between the origin and target network switches.
    """
    topology = origin.model.topology
    path = []

    if not hasattr(topology, "delay_shortest_paths"):
        topology.delay_shortest_paths = {}

    key = (origin, target)

    if key in topology.delay_shortest_paths.keys():
        path = topology.delay_shortest_paths[key]
    else:
        path = nx.shortest_path(topology, source=origin, target=target, weight="delay")
        topology.delay_shortest_paths[key] = path

    return path


def normalize_cpu_and_memory(cpu, memory) -> float:
    """Normalizes the CPU and memory values.

    Args:
        cpu (float): CPU value.
        memory (float): Memory value.

    Returns:
        normalized_value (float): Normalized value.
    """
    normalized_value = (cpu * memory) ** (1 / 2)
    return normalized_value


def application_has_useful_migrations_for_user(application: edge_sim_py.Application, user: edge_sim_py.User) -> bool:
    """Method that checks if an application has useful migrations. (i.e., at the end of the ongoing migrations, the delay SLA is met)
    Args:
        application (edge_sim_py.Application): Application object.
    Returns:
        bool: True if the application has useful migrations, False otherwise.
    """

    services_base_stations = [
        service._Service__migrations[-1]["target"].base_station
        if len(service._Service__migrations) > 0 and service._Service__migrations[-1]["status"] != "finished"
        else service.server.base_station
    for service in application.services]

    base_stations = [user.base_station, *services_base_stations]

    delay = base_stations[0].wireless_delay
    for i in range(1, len(base_stations)):
        path = find_shortest_path(base_stations[i - 1].network_switch, target=base_stations[i].network_switch)
        path_delay = sum([edge_sim_py.Topology.first()[path[i]][path[i + 1]]["delay"] for i in range(len(path) - 1)])
        delay += path_delay

    if delay > user.delay_slas[str(application.id)]:
        return False

    return True


def edge_server_step_with_least_congested_shortest_path(self):
    """Method that executes the events involving the object at each time step.
    
    The difference from EdgeSimPy's default method is that this custom method uses the least congested shortest path 
    instead of simply the shortest path to select the network path to be used to pull the container image from the registry.
    """
    while len(self.waiting_queue) > 0 and len(self.download_queue) < self.max_concurrent_layer_downloads:
        layer = self.waiting_queue.pop(0)

        # Gathering the list of registries that have the layer
        registries_with_layer = []
        for registry in [reg for reg in edge_sim_py.ContainerRegistry.all() if reg.available]:
            # Checking if the registry is hosted on a valid host in the infrastructure and if it has the layer we need to pull
            if registry.server and any(layer.digest == l.digest for l in registry.server.container_layers):
                # Selecting a network path to be used to pull the layer from the registry
                path_data = least_congested_shortest_path(
                    self.model.topology,
                    registry.server.network_switch,
                    self.network_switch
                )

                registries_with_layer.append({
                    "object": registry,
                    "path": path_data["path"],
                    "max_data_to_transfer": path_data["max_data_to_transfer"]
                })

        # Selecting the registry from which the layer will be pulled to the (target) edge server
        registries_with_layer = sorted(registries_with_layer, key=lambda r: (len(r["path"]), r["max_data_to_transfer"]))
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
            metadata={"type": "layer", "object": layer, "container_registry": registry, "size": layer.size},
        )
        flow.total_size=layer.size
        self.model.initialize_agent(agent=flow)

        # Adding the created flow to the edge server's download queue
        self.download_queue.append(flow)


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
        "Data to Transfer": self.data_to_transfer,
        "Data Size": self.metadata["size"],
    }
    return metrics


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

            # Defining origin and target nodes
            origin = communication_chain[i]
            target = communication_chain[i + 1]

            # Finding and storing the best communication path between the origin and target nodes
            if origin == target:
                path = []
            else:
                path = find_shortest_path(origin=origin.network_switch, target=target.network_switch)

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
    # Checking for delay SLA violations
    for app in self.applications:
        if self.delays[str(app.id)] != None and self.delays[str(app.id)] != float("inf") and self.delays[str(app.id)] > self.delay_slas[str(app.id)]:
            self.delay_sla_violations[str(app.id)] += 1
            self.accumulated_violation_intensity += self.delays[str(app.id)] - self.delay_slas[str(app.id)]
            # Checking for delay SLA violations during migrations
            is_violation_during_migration = False
            for service in app.services:
                if len(service._Service__migrations) > 0 and service._Service__migrations[-1]["status"] != "finished":
                    is_violation_during_migration = True
            
            if is_violation_during_migration:
                self.delay_sla_violations_during_migrations[str(app.id)] += 1

    # Checking for useless migrations
    self.steps_during_useless_migrations += 0 if application_has_useful_migrations_for_user(application=self.applications[0], user=self) else 1

    # Computing application metrics
    application_chain_size = len(self.applications[0].services)
    application_cpu_demand = sum([service.cpu_demand for service in self.applications[0].services])
    application_memory_demand = sum([service.memory_demand for service in self.applications[0].services])

    metrics = {
        "Instance ID": self.id,
        "Coordinates": self.coordinates,
        "Base Station": f"{self.base_station} ({self.base_station.coordinates})" if self.base_station else None,
        "Delays": sum(self.delays.values()),
        "Delay SLAs": sum(self.delay_slas.values()),
        "Delay SLA Violations": sum(self.delay_sla_violations.values()),
        "Delay SLA Violations During Migrations": sum(self.delay_sla_violations_during_migrations.values()),
        "Application Chain Size": application_chain_size,
        "Application CPU Demand": application_cpu_demand,
        "Application Memory Demand": application_memory_demand,
        "Steps During Useless Migrations": self.steps_during_useless_migrations,
        "Accumulated Violation Intensity": self.accumulated_violation_intensity,
    }
    return metrics


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
