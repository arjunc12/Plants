import networkx as nx
from random_graphs import random_mst
from itertools import combinations
import numpy as np
from cost_functions import *
import pylab
from bisect import bisect_left, insort
from random import sample, choice, uniform, randint, random, seed
from collections import defaultdict
from random_graphs import random_point_graph
from steiner_midpoint import *
from sys import argv
from scipy.spatial.distance import euclidean
from graph_utils import *
from time import time
import argparse

STEINER_MIDPOINTS = 10

def satellite_tree(G):
    root = G.graph['root']
    
    satellite = G.copy()
    satellite.remove_edges_from(G.edges())

    for u in satellite.nodes():
        if u != root:
            satellite.add_edge(u, root)
            p1, p2 = satellite.node[u]['coord'], satellite.node[root]['coord']
            satellite[u][root]['length'] = point_dist(p1, p2)
    
    return satellite

def min_spanning_tree(G):
    return nx.minimum_spanning_tree(G, weight='length')

def pareto_steiner(G, alpha):
    root = G.graph['root']

    H = nx.Graph()
  
    H.add_node(root)
    H.graph['root'] = root
    H.node[root]['droot'] = 0
    H.node[root]['parent'] = None
    root_coord = G.node[root]['coord']
    H.node[root]['coord'] = root_coord
    H.node[root]['label'] = 'root'
    added_nodes = 1

    in_nodes = set()
    out_nodes = set(G.nodes())
    in_nodes.add(root)
    out_nodes.remove(root)
   
    graph_mcost = 0
    graph_scost = 0

    closest_neighbors = {}
    is_sorted = 'sorted' in G.graph
    for u in G.nodes():
        if is_sorted:
            closest_neighbors[u] = G.node[u]['close_neighbors'][:]
        else:
            closest_neighbors[u] = k_nearest_neighbors(G, u, k=None, candidate_nodes=None)

    unpaired_nodes = set([root])

    node_index = G.number_of_nodes() + 1

    dist_error = 0

    steps = 0

    best_edges = []
    while added_nodes < G.number_of_nodes():
        assert len(out_nodes) > 0
        best_edge = None
        best_mcost = None
        best_scost = None
        best_cost = float("inf")

        best_choice = None
        best_midpoint = None

        candidate_edges = []
        for u in unpaired_nodes:
            assert H.has_node(u)
            assert 'droot' in H.node[u]
        
            invalid_neighbors = []
            closest_neighbor = None
            for i in xrange(len(closest_neighbors[u])):
                v = closest_neighbors[u][i]
                if H.has_node(v):
                    invalid_neighbors.append(v)
                else:
                    closest_neighbor = v
                    break

            for invalid_neighbor in invalid_neighbors:
                closest_neighbors[u].remove(invalid_neighbor)
           
            assert closest_neighbor != None
            assert not H.has_node(closest_neighbor)
            
            p1 = H.node[u]['coord']
            p2 = G.node[closest_neighbor]['coord']
                
            length = point_dist(p1, p2)
            mcost = length
            scost = length + H.node[u]['droot']
            cost = pareto_cost(mcost=mcost, scost=scost, alpha=alpha)
            insort(best_edges, (cost, u, closest_neighbor))

        cost, u, v = best_edges.pop(0)

        best_edges2 = []
        unpaired_nodes = set([u, v])
        for cost, x, y in best_edges:
            if y == v:
                unpaired_nodes.add(x)
            else:
                best_edges2.append((cost, x, y))
        best_edges = best_edges2

        assert H.has_node(u)
        assert not H.has_node(v)
        H.add_node(v)
        H.node[v]['coord'] = G.node[v]['coord']
        H.node[v]['label'] = 'synapse'
        in_nodes.add(v)
        out_nodes.remove(v)

        p1 = H.node[u]['coord']
        p2 = H.node[v]['coord']
        midpoints = steiner_points(p1, p2, npoints=STEINER_MIDPOINTS)
        midpoint_nodes = []
        for midpoint in midpoints:
            midpoint_node = 'b%d' % node_index
            node_index += 1
            H.add_node(midpoint_node)
            H.node[midpoint_node]['coord'] = midpoint

            neighbors = []
            for out_node in out_nodes:
                out_coord = G.node[out_node]['coord']
                dist = point_dist(midpoint, out_coord)
                neighbors.append((dist, out_node))

            neighbors = sorted(neighbors)
            closest_neighbors[midpoint_node] = []
            for dist, neighbor in neighbors:
                closest_neighbors[midpoint_node].append(neighbor)

            midpoint_nodes.append(midpoint_node)

            unpaired_nodes.add(midpoint_node)

        line_nodes = [v] + list(reversed(midpoint_nodes)) + [u]
        for i in xrange(-1, -len(line_nodes), -1):
            n1 = line_nodes[i]
            n2 = line_nodes[i - 1]
            H.add_edge(n1, n2)
            H[n1][n2]['length'] = node_dist(H, n1, n2)
            assert 'droot' in H.node[n1]
            H.node[n2]['parent'] = n1
            H.node[n2]['droot'] = node_dist(H, n2, u) + H.node[u]['droot']
            if not G.has_node(n2):
                H.node[n2]['label'] = 'steiner_midpoint'

        added_nodes += 1
    return H 

def pareto_brute_force(G, alpha, trees=None):
    if trees == None:
        trees = find_all_spanning_trees(G)

    best_tree = None
    best_cost = float("inf")
    for tree in trees:
        mcost, scost = graph_costs(tree)
        cost = pareto_cost(mcost, scost, alpha)
        if cost < best_cost:
            best_cost = cost
            best_tree = tree

    return best_tree

def centroid(G):
    root = G.graph['root']
    root_coord = G.node[root]['coord']
    centroid = np.zeros(len(root_coord))
    for u in G.nodes():
        point = G.node[u]['coord']
        assert len(point) == len(root_coord)
        if u != root:
            centroid += point
    centroid /= G.number_of_nodes() - 1
    return centroid

def centroid_mst(G):
    cent_mst = G.copy()
    cent_mst.remove_edges_from(G.edges())
    
    centroidp = centroid(G)
    cent_mst.add_node('centroid')
    cent_mst.node['centroid']['label'] = 'centroid'
    cent_mst.node['centroid']['coord'] = centroidp
    for u in G.nodes():
        cent_mst.add_edge(u, 'centroid')
        cent_mst[u]['centroid']['length'] = point_dist(cent_mst.node[u]['coord'], centroidp)
    return cent_mst

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--points', type=int, default=100)
    parser.add_argument('-a', '--alpha', type=float, default=0.5)

    args = parser.parse_args()
    points = args.points
    alpha = args.alpha

    G = random_point_graph(points)

    #algorithms = [pareto_steiner_space, pareto_steiner_space2, pareto_steiner_fast, pareto_steiner_old]
    algorithms = [pareto_steiner_fast, pareto_prim]
    for pareto_func in algorithms:
        start = time()
        tree = pareto_func(G, alpha)
        end = time()
        print graph_costs(tree, relevant_nodes=G.nodes())
        print end - start

if __name__ == '__main__':
   main()
