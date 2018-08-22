import networkx as nx
from dist_functions import node_dist
import os

DATA_DIR = '/iblsn/data/Arjun/Plants'
DATASETS_DIR = '%s/datasets' % DATA_DIR

LABELS = {'r' : 'root', 's' : 'stem', 'c' : 'cotyledon', 'b' : 'branch',\
              'l' : 'leaf'}

INPUT_LABELS = ['root', 'cotyledon', 'leaf']

def parse_node(G, line):
    line = line.split(',')
    assert len(line) == 4
    node = line[0]
    node = node.strip('\"')
    node = node.strip(' -nom-')
    G.add_node(node)
    G.node[node]['coord'] = map(float, line[1:])
    
    label = LABELS[node[0]]
    G.node[node]['component'] = label
    if label in INPUT_LABELS:
        G.graph['input points'].append(node)
    if label == 'root':
        G.graph['root'] = node

def parse_edge(G, line):
    parent, children = line.split(':')
    parent = parent.strip()
    assert G.has_node(parent)
    children = children.split(',')
    for child in children:
        child = child.strip()
        assert G.has_node(child)
        G.add_edge(parent, child)
        G[parent][child]['length'] = node_dist(G, parent, child)

def read_tree(fname):
    G = nx.Graph()
    G.graph['input points'] = []
    edges = False
    with open(fname) as f:
        for line in f:
            line = line.strip('\n')
            line = line.strip('\r')
            line = line.strip()

            if line == '' or line == '#end':
                continue
            elif line == '#edges':
                edges = True
                continue

            if edges:
                parse_edge(G, line)
            else:
                parse_node(G, line)
    return G

def main():
    for fname in os.listdir(PLANTS_DIR):
        print fname
        G = read_tree('%s/%s' % (PLANTS_DIR, fname))
        print G.number_of_nodes()

if __name__ == '__main__':
    main()
