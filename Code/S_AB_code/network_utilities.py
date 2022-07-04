import networkx, random, copy

def create_graph(directed=False):
    """
        Creates & returns a graph
    """
    if directed:
        g = networkx.DiGraph()
    else:
        g = networkx.Graph()
    return g


def get_nodes_and_edges_from_sif_file(file_name, store_edge_type = False, delim=None, data_to_float=True):
    
    setNode = set()
    setEdge = set()
    dictNode = {}
    dictEdge = {}
    flag = False
    f=open(file_name)
    for line in f:
        if delim is None:
            words = line.rstrip("\n").split()
        else:
            words = line.rstrip("\n").split(delim)
        id1 = words[0]
        setNode.add(id1)
        if len(words) == 2:
            if data_to_float:
                score = float(words[1])
            else:
                score = words[1]
            dictNode[id1] = score
        elif len(words) >= 3: 
            if len(words) > 3:
                flag = True
            id2 = words[2]
            setNode.add(id2)
            setEdge.add((id1, id2))
            if store_edge_type:
                if data_to_float:
                    dictEdge[(id1, id2)] = float(words[1])
                else:
                    dictEdge[(id1, id2)] = words[1]
    f.close()
    if len(setEdge) == 0:
        setEdge = None
    if len(dictNode) == 0:
        dictNode = None
    if len(dictEdge) == 0:
        dictEdge = None
    if flag:
        print("Warning: Ignored extra columns in the file!")
    return setNode, setEdge, dictNode, dictEdge



def create_network_from_sif_file(network_file_in_sif, use_edge_data = False, delim = None, include_unconnected=True):
    setNode, setEdge, dictDummy, dictEdge = get_nodes_and_edges_from_sif_file(network_file_in_sif, store_edge_type = use_edge_data, delim = delim)
    g = create_graph()
    if include_unconnected:
        g.add_nodes_from(setNode)
    if use_edge_data:
        for e,w in dictEdge.iteritems():
        	u,v = e
        	g.add_edge(u,v,w=w) #,{'w':w})
    else:
        g.add_edges_from(setEdge)
    return g


def get_connected_components(G, return_as_graph_list=True):
    result_list = []

    if return_as_graph_list:
        result_list = networkx.connected_component_subgraphs(G)
    else:
        result_list = [c for c in sorted(networkx.connected_components(G), key=len, reverse=True)]
    return result_list


def get_subgraph(G, nodes):
    return G.subgraph(nodes)


def get_shortest_path_length_between(G, source_id, target_id):
    return networkx.shortest_path_length(G, source_id, target_id)


def get_degree_binning(g, bin_size, lengths=None):
    degree_to_nodes = {}
    for node, degree in g.degree(): #.iteritems(): # iterator in networkx 2.0
        if lengths is not None and node not in lengths:
            continue
        degree_to_nodes.setdefault(degree, []).append(node)
    values = degree_to_nodes.keys()
    values = sorted(values)
    bins = []
    i = 0
    while i < len(values):
        low = values[i]
        val = degree_to_nodes[values[i]]
        while len(val) < bin_size:
            i += 1
            if i == len(values):
                break
            val.extend(degree_to_nodes[values[i]])
        if i == len(values):
            i -= 1
        high = values[i]
        i += 1  
        if len(val) < bin_size:
            low_, high_, val_ = bins[-1]
            bins[-1] = (low_, high, val_ + val)
        else:
            bins.append((low, high, val))
    return bins


def pick_random_nodes_matching_selected(network, bins, nodes_selected, n_random, degree_aware=True, connected=False, seed=None):
    """
    Use get_degree_binning to get bins
    """
    if seed is not None:
        random.seed(seed)
    values = []
    nodes = network.nodes()
    for i in range(n_random):
        if degree_aware:
            if connected:
                raise ValueError("Not implemented!")
            nodes_random = set()
            node_to_equivalent_nodes = get_degree_equivalents(nodes_selected, bins, network)
            for node, equivalent_nodes in node_to_equivalent_nodes.items():
                chosen = random.choice(equivalent_nodes)
                for k in range(20): # Try to find a distinct node (at most 20 times)
                    if chosen in nodes_random:
                        chosen = random.choice(equivalent_nodes)
                nodes_random.add(chosen)
            nodes_random = list(nodes_random)
        else:
            if connected:
                nodes_random = [ random.choice(nodes) ]
                k = 1
                while True:
                    if k == len(nodes_selected):
                        break
                    node_random = random.choice(nodes_random)
                    node_selected = random.choice(network.neighbors(node_random))
                    if node_selected in nodes_random:
                        continue
                    nodes_random.append(node_selected)
                    k += 1
            else:
                nodes_random = random.sample(nodes, len(nodes_selected))
        values.append(nodes_random)
    return values


def get_degree_equivalents(seeds, bins, g):
    seed_to_nodes = {}
    for seed in seeds:
        d = g.degree(seed)
        for l, h, nodes in bins:
            if l <= d and h >= d:
                mod_nodes = list(nodes)
                mod_nodes.remove(seed)
                seed_to_nodes[seed] = mod_nodes
                break
    return seed_to_nodes