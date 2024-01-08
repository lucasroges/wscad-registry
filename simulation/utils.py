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


def find_shortest_path(source: edge_sim_py.NetworkSwitch, target: edge_sim_py.NetworkSwitch) -> list:
    """Finds the shortest path (delay used as weight) between two network switches (source and target).

    Args:
        source (edge_sim_py.NetworkSwitch): Source network switch.
        target (edge_sim_py.NetworkSwitch): Target network switch.

    Returns:
        path (list): Shortest path between the source and target network switches.
    """
    topology = source.model.topology
    path = []

    if not hasattr(topology, "delay_shortest_paths"):
        topology.delay_shortest_paths = {}

    key = (source, target)

    if key in topology.delay_shortest_paths.keys():
        path = topology.delay_shortest_paths[key]
    else:
        path = nx.shortest_path(topology, source=source, target=target, weight="delay")
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

