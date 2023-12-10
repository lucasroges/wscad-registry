import math
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


def follow_user():
    """Simple strategy that keeps application services as close as possible to each other and to the end-user.
    However, this adapted strategy only migrates when the delay is greater than or equal to the delay SLA.
    It migrates user's services to the edge server closest to the base station used by the user or the previous service in the app chain
    (we use network delay as proximity measure, as shown in the sorting process from get_candidate_hosts function).
    """
    # Iterating over all users
    for user in edge_sim_py.User.all():
        applications = user.applications

        for application in applications:
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


def least_congested_shortest_path(topology: edge_sim_py.Topology, source: edge_sim_py.NetworkSwitch, target: edge_sim_py.NetworkSwitch) -> list:
    """Finds the least congested shortest path between two network switches.

    Args:
        topology (edge_sim_py.Topology): Network topology
        source (edge_sim_py.NetworkSwitch): Source network switch
        target (edge_sim_py.NetworkSwitch): Target network switch

    Returns:
        list: Least congested shortest path between the source and target network switches    
    """
    # TODO: replace this method with an utils method similar to 'find_shortest_path' (e.g., find_all_shortest_paths)
    # However, this change might not be enough to decrease the simulation time, as the lines below have a significant cost
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


def get_geometric_mean(values: list):
    """Calculates the geometric mean of a list of values.

    Args:
        values (list): List of values.

    Returns:
        geometric_mean (float): Geometric mean of the list of values.
    """
    number_of_values = len(values)
    geometric_mean = math.prod(values) ** (1 / number_of_values)
    return geometric_mean


def get_mean_latency(simulator: edge_sim_py.Simulator):
    """Calculates the mean latency of the simulation.

    Args:
        simulator (edge_sim_py.Simulator): Simulation object.

    Returns:
        mean_latency (float): Mean latency of the simulation.
    """
    user_metrics = simulator.agent_metrics["User"]
    accumulated_latency = sum([user_metric["Delays"] for user_metric in user_metrics])
    mean_latency = accumulated_latency / len(user_metrics)

    return mean_latency


def get_mean_provisioning_time(simulator: edge_sim_py.Simulator):
    """Calculates the mean provisioning time of the simulation.

    Args:
        simulator (edge_sim_py.Simulator): Simulation object.

    Returns:
        mean_provisioning_time (float): Mean provisioning time of the simulation.
    """
    service_metrics_from_last_step = [service_metric for service_metric in simulator.agent_metrics["Service"] if service_metric["Time Step"] == simulator.schedule.steps]
    accumulated_provisioning_time = sum([service_metric["Average Migration Duration"] for service_metric in service_metrics_from_last_step])
    mean_provisioning_time = accumulated_provisioning_time / len(service_metrics_from_last_step)

    return mean_provisioning_time


def get_overloaded_edge_servers(simulator: edge_sim_py.Simulator):
    """Calculates the number of overloaded edge servers in the simulation.

    Args:
        simulator (edge_sim_py.Simulator): Simulation object.

    Returns:
        overloaded_edge_servers (int): Accumulated number of overloaded edge servers in the simulation.
    """
    topology_metrics = simulator.agent_metrics["Topology"]
    accumulated_overloaded_edge_servers = sum([topology_metric["Overloaded Edge Servers"] for topology_metric in topology_metrics])

    return accumulated_overloaded_edge_servers