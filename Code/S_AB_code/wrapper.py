import network_utilities
import csv, numpy, os
import random

#####======================================= Network related ===================================================#####

def get_network(network_file, only_lcc):
    network = network_utilities.create_network_from_sif_file(network_file, use_edge_data = False, delim = None, include_unconnected=True)
    #print len(network.nodes()), len(network.edges())
    if only_lcc and not network_file.endswith(".lcc"):
        components = network_utilities.get_connected_components(network, False)
        network = network_utilities.get_subgraph(network, components[0])
        network_lcc_file = network_file + ".lcc"
        if not os.path.exists(network_lcc_file ):
            f = open(network_lcc_file, 'w')
            for u,v in network.edges():
                f.write("%s 1 %s\n" % (u, v))
            f.close()
    return network


###============================================= Separation related ====================================================


def calculate_separation_proximity(network, nodes_from, nodes_to, nodes_from_random=None, nodes_to_random=None, bins=None, n_random=1000, min_bin_size=100, seed=452456, lengths=None):
    """
    Calculate proximity from nodes_from to nodes_to
    If degree binning or random nodes are not given, they are generated
    lengths: precalculated shortest path length dictionary
    """
    nodes_network = set(network.nodes())
    if len(set(nodes_from) & nodes_network) == 0 or len(set(nodes_to) & nodes_network) == 0:
        return None # At least one of the node group not in network
    d = get_separation(network, nodes_from, nodes_to, lengths)
    return d


def get_separation(network, nodes_from, nodes_to, lengths=None):
    dAA = numpy.mean(get_separation_within_set(network, nodes_from, lengths))
    dBB = numpy.mean(get_separation_within_set(network, nodes_to, lengths))
    dAB = numpy.mean(get_separation_between_sets(network, nodes_from, nodes_to, lengths))
    d = dAB - (dAA + dBB) / 2.0
    return d


def get_separation_between_sets(network, nodes_from, nodes_to, lengths=None):
    """
    Calculate dAB in separation metric proposed by Menche et al. 2015
    """
    values = []
    target_to_values = {}
    source_to_values = {}
    for source_id in nodes_from:
        for target_id in nodes_to:
            if lengths is not None:
                d = lengths[source_id][target_id] 
            else:
                d = network_utilities.get_shortest_path_length_between(network, source_id, target_id)
            source_to_values.setdefault(source_id, []).append(d)
            target_to_values.setdefault(target_id, []).append(d)
    # Distances to closest node in nodes_to (B) from nodes_from (A)
    for source_id in nodes_from:
        inner_values = source_to_values[source_id]
        values.append(numpy.min(inner_values))
    # Distances to closest node in nodes_from (A) from nodes_to (B)
    for target_id in nodes_to:
        inner_values = target_to_values[target_id]
        values.append(numpy.min(inner_values))
    return values


def get_separation_within_set(network, nodes_from, lengths=None):
    """
    Calculate dAA or dBB in separation metric proposed by Menche et al. 2015
    """
    if len(nodes_from) == 1:
        return [ 0 ]
    values = []
    # Distance to closest node within the set (A or B)
    for source_id in nodes_from:
        inner_values = []
        for target_id in nodes_from:
            if source_id == target_id:
                continue
            if lengths is not None:
                d = lengths[source_id][target_id] 
            else:
                d = network_utilities.get_shortest_path_length_between(network, source_id, target_id)
            inner_values.append(d)
        print("inner")
        print(inner_values)
        values.append(numpy.min(inner_values))
    print("values")
    print(values)
    return values




###============================================= Proximity related ====================================================

def calculate_proximity(network, nodes_from, nodes_to, nodes_from_random=None, nodes_to_random=None, bins=None, n_random=1000, min_bin_size=100, seed=452456, lengths=None):
    """
    Calculate proximity from nodes_from to nodes_to
    If degree binning or random nodes are not given, they are generated
    lengths: precalculated shortest path length dictionary
    """
    #distance = "closest"
    #lengths = network_utilities.get_shortest_path_lengths(network, "../data/toy.sif.pcl")
    #d = network_utilities.get_separation(network, lengths, nodes_from, nodes_to, distance, parameters = {})

    nodes_network = set(network.nodes())
    nodes_from = set(nodes_from) & nodes_network 
    nodes_to = set(nodes_to) & nodes_network
    if len(nodes_from) == 0 or len(nodes_to) == 0:
        return None
    d = calculate_closest_distance(network, nodes_from, nodes_to, lengths)
    print("--------------------d--------------")
    print(d)
    if bins is None and (nodes_from_random is None or nodes_to_random is None):
        bins = network_utilities.get_degree_binning(network, min_bin_size, lengths) # if lengths is given, it will only use those nodes
    print("--------------------bins--------------")
    print(bins)
    if nodes_from_random is None:
        nodes_from_random = get_random_nodes(nodes_from, network, bins = bins, n_random = n_random, min_bin_size = min_bin_size, seed = seed)
    print("--------------------nodes_from_random--------------")
    print(nodes_from_random)
    if nodes_to_random is None:
        nodes_to_random = get_random_nodes(nodes_to, network, bins = bins, n_random = n_random, min_bin_size = min_bin_size, seed = seed)
    print("--------------------nodes_to_random--------------")
    print(nodes_to_random)
    random_values_list = zip(nodes_from_random, nodes_to_random)
    values = numpy.empty(len(nodes_from_random)) #n_random
    print(len(values))
    print("---------------------count------------------")
    for i, values_random in enumerate(random_values_list):
        print(i)
        nodes_from, nodes_to = values_random
        values[i] = calculate_closest_distance(network, nodes_from, nodes_to, lengths)
    m, s  = numpy.mean(values), numpy.std(values)
    if s == 0:
        z = 0.0
    else:
        z = (d - m) / s
    return d, z, (m, s)



def calculate_closest_distance(network, nodes_from, nodes_to, lengths=None):
    values_outer = []
    if lengths is None:
        for node_from in nodes_from:
            values = []
            for node_to in nodes_to:
                val = network_utilities.get_shortest_path_length_between(network, node_from, node_to)
                values.append(val)
            d = min(values)
            values_outer.append(d)
    else:
        for node_from in nodes_from:
            values = []
            vals = lengths[node_from]
            for node_to in nodes_to:
                val = vals[node_to]
                values.append(val)
            d = min(values)
            values_outer.append(d)
    d = numpy.mean(values_outer)
    return d


def get_random_nodes(nodes, network, bins=None, n_random=1000, min_bin_size=100, degree_aware=True, seed=None):
    if bins is None:
        bins = network_utilities.get_degree_binning(network, min_bin_size) 
    nodes_random = network_utilities.pick_random_nodes_matching_selected(network, bins, nodes, n_random, degree_aware, seed=seed) 
    return nodes_random