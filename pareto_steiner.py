from time import time, sleep
import numpy as np
import networkx as nx
from sys import argv
import matplotlib as mpl
mpl.use('agg')
import pylab
import os
from random import shuffle
from itertools import combinations
import argparse
from cost_functions import *
from pareto_functions import *
from random_graphs import random_mst, barabasi_tree
import math
from collections import defaultdict
from numpy.random import permutation
from tradeoff_ratio import tradeoff_ratio
from plant_utils import *

VIZ_TREES = True

LOG_PLOT = True

DATA_DRIVE = '/iblsn/data/Arjun/Plants'

DATASETS_DIR = '%s/datasets' % DATA_DRIVE
FIGS_DIR = '%s/pareto_steiner_output/steiner_figs' % DATA_DRIVE
OUTPUT_DIR = '%s/pareto_steiner_output' % DATA_DRIVE
FRONTS_DIR = '%s/pareto_fronts' % OUTPUT_DIR
TEMP_DIR = '%s/pareto_steiner_temp' % OUTPUT_DIR
PLOTS_DIR = '%s/pareto_front_plots' % OUTPUT_DIR
LOG_PLOTS_DIR = '%s/pareto_front_log_plots' % OUTPUT_DIR
DIST_FUNC = pareto_dist_scale


COLORS = {'plant' : 'r', 'centroid' : 'g', 'random' : 'm', 'barabasi' : 'c'}
MARKERS = {'plant' : 'x', 'centroid' : 'o', 'random' : '^', 'barabasi' : 's'}
LABELS = {'plant' : 'Plant arbor', 'centroid' : 'Centroid', 'random' : 'Random', 'barabasi' : 'Barabasi-Albert'}

PLOT_TREES = ['plant']
LOG_PLOT_TREES = ['plant', 'centroid', 'barabasi', 'random']
    
ARBOR_TYPES = {'axon': 0, 'basal_dendrite' : 1, 'apical_dendrite' : 2, 'truncated_axon' : 3}

def ceil_power_of_10(n):
    exp = math.log(n, 10)
    exp = math.ceil(exp)
    return 10**exp

def floor_power_of_10(n):
    exp = math.log(n, 10)
    exp = math.ceil(exp)
    return 10**exp

def read_pareto_front(fronts_dir):
    alphas = []
    mcosts = []
    scosts = []
    with open('%s/pareto_front.csv' % fronts_dir) as pareto_front:
        for line in pareto_front:
            line = line.strip('\n')
            line = line.split(', ')
            if line[0] == 'alpha':
                continue
            alpha = float(line[0])
            mcost = float(line[1])
            scost = float(line[2])

            alphas.append(alpha)
            mcosts.append(mcost)
            scosts.append(scost)

    return alphas, mcosts, scosts

def read_tree_costs(fronts_dir):
    tree_costs = defaultdict(lambda : defaultdict(list))
    with open('%s/tree_costs.csv' % fronts_dir) as tree_costs_file:
        for line in tree_costs_file:
            line = line.strip('\n')
            line = line.split(', ')
            if line[0] == 'tree':
                continue
            model = line[0]
            mcost = float(line[1])
            scost = float(line[2])
            tree_costs[model]['mcost'].append(mcost)
            tree_costs[model]['scost'].append(scost)

    return tree_costs

def pareto_plot(fronts_dir, figs_dir, log_plot=False,\
                plant=None):
    import seaborn as sns
    '''
    pareto_front = pd.read_csv('%s/pareto_front.csv' % fronts_dir,\
                               skipinitialspace=True)
    mcosts = pareto_front['mcost']
    scosts = pareto_front['scost']
    '''
    alphas, mcosts, scosts = read_pareto_front(fronts_dir)
    
    alphas = alphas[1:]
    mcosts = mcosts[1:]
    scosts = scosts[1:]

    if log_plot:
        mcosts = pylab.log10(pylab.array(mcosts))
        scosts = pylab.log10(pylab.array(scosts))

    '''
    tree_costs = pd.read_csv('%s/tree_costs.csv' % fronts_dir,\
                             skipinitialspace=True)
    '''
    tree_costs = read_tree_costs(fronts_dir)

    pylab.figure()
    sns.set()

    pylab.plot(mcosts, scosts, c='b', label='_nolegend_')
    pylab.scatter(mcosts, scosts, c='b', label='Pareto front')
 
    plot_trees = None
    if log_plot:
        plot_trees = LOG_PLOT_TREES
    else:
        plot_trees = PLOT_TREES
    #for tree, costs in tree_costs.groupby('tree'):
    for tree, costs in tree_costs.iteritems():
        if tree in plot_trees:
            mcosts = costs['mcost']
            scosts = costs['scost']
            if log_plot:
                mcosts = pylab.log10(pylab.array(mcosts))
                scosts = pylab.log10(pylab.array(scosts))
            pylab.scatter(mcosts, scosts, label=LABELS[tree],\
                          marker=MARKERS[tree], s=175, c=COLORS[tree])
    
    xlab = 'Wiring Cost'
    ylab = 'Conduction Delay'
    if log_plot:
        xlab = 'log(' + xlab + ')'
        ylab = 'log(' + ylab + ')'
    
    pylab.xlabel(xlab, fontsize=35)
    pylab.ylabel(ylab, fontsize=35)
    leg = pylab.legend(frameon=True)
    
    ax = pylab.gca()
    
    pylab.setp(ax.get_legend().get_texts(), fontsize=30) # for legend text
    
    leg.get_frame().set_linewidth(5)
    leg.get_frame().set_edgecolor('k')
    
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    
    pylab.tight_layout()
   
    pdf_name = ''
    if log_plot:
        pdf_name += 'log_'
    pdf_name += 'pareto_front.pdf'
    fname = '%s/%s' % (figs_dir, pdf_name)
    pylab.savefig(fname, format='pdf')

    if neuron_name != None and neuron_type != None:
        plot_dir = None
        
        if log_plot:
            plot_dir = LOG_PLOTS_DIR
        else:
            plot_dir = PLOTS_DIR
        
        neuron_name = neuron_name.replace(' ', '_')
        neuron_type = neuron_type.replace(' ', '_')
        plot_dir = '%s/%s/%s' % (plot_dir, neuron_name, neuron_type)
        os.system('mkdir -p %s' % plot_dir)
        fname = '%s/%s' % (plot_dir, pdf_name)
        pylab.savefig(fname, format='pdf')

    pylab.close()

def pareto_tree_costs(G, point_graph, viz_trees=False, figs_dir=None,\
                      sandbox=False, log_dir=None):
    delta = 0.01
    alphas = np.arange(delta, 1 + delta, delta)
    mcosts = []
    scosts = []

    pareto_func = pareto_steiner
    if sandbox:
        pareto_func = pareto_steiner_sandbox
    total_time = 0
    log_fname = '%s/logging.txt' % log_dir
    for i, alpha in enumerate(alphas):
        print alpha
        start = time()
        pareto_tree = pareto_func(point_graph, alpha)
        end = time()
        t = (end - start) / 60.0
        log_file = open(log_fname, 'a')
        log_file.write('%f, %f\n' % (alpha, t))
        log_file.close()

        total_time += t
        mcost = mst_cost(pareto_tree)
        scost = satellite_cost(pareto_tree, relevant_nodes=point_graph.nodes())
        cost = pareto_cost(mcost=mcost, scost=scost, alpha=alpha)

        if (i % 5 == 4) and viz_trees:
            assert figs_dir != None
            viz_tree(pareto_tree, '%s-%0.2f' % ('alpha', alpha), outdir=figs_dir)
        
        mcosts.append(mcost)
        scosts.append(scost)

    log_file = open(log_fname, 'a')
    log_file.write('%f\n' % total_time)
    log_file.close()
    return alphas, mcosts, scosts

def pareto_front(G, point_graph, fronts_dir=FRONTS_DIR, figs_dir=FIGS_DIR,\
                 viz_trees=VIZ_TREES, sandbox=False):

    sat_tree = satellite_tree(point_graph)
    sat_scost = satellite_cost(sat_tree)
    sat_mcost = mst_cost(sat_tree)
    span_tree = nx.minimum_spanning_tree(point_graph, weight='length') 

# ---------------------------------------
    pareto_front_fname = '%s/pareto_front.csv' % fronts_dir
    first_time = not os.path.exists(pareto_front_fname)

# ---------------------------------------
    alphas = None
    mcosts = None
    scosts = None

# ---------------------------------------
    if (not first_time) and (not viz_trees):
        alphas, mcosts, scosts = read_pareto_front(fronts_dir)
    else:
        front_lines = []
        front_lines.append('alpha, mcost, scost\n')
        front_lines.append('%f, %f, %f\n' % (0, sat_mcost, sat_scost))

        log_dir = fronts_dir
        alphas, mcosts, scosts = pareto_tree_costs(G, point_graph,\
                                                   viz_trees=viz_trees,\
                                                   figs_dir=figs_dir,\
                                                   sandbox=sandbox,\
                                                   log_dir=log_dir)

        for i in xrange(len(alphas)):
            alpha = alphas[i]
            mcost = mcosts[i]
            scost = scosts[i]
            front_lines.append('%f, %f, %f\n' % (alpha, mcost, scost))

        front_fname = '%s/pareto_front.csv' % fronts_dir
        with open(front_fname, 'w') as front_file:
            front_file.writelines(front_lines)
    
# ---------------------------------------
    for u in point_graph.nodes():
        for H in [G, sat_tree, span_tree]:
            assert H.has_node(u)
            if u == G.graph['root']:
                H.node[u]['label'] = 'root'
            else:
                H.node[u]['label'] = 'synapse'
    if viz_trees:
        viz_tree(G, 'neural', outdir=figs_dir) 
        viz_tree(sat_tree, 'sat', outdir=figs_dir)
        viz_tree(span_tree, 'mst', outdir=figs_dir)

# ---------------------------------------
    return alphas, mcosts, scosts, first_time

def pareto_analysis(G, plant, fronts_dir=FRONTS_DIR, output_dir=OUTPUT_DIR,\
                    figs_dir=FIGS_DIR, output=True, viz_trees=VIZ_TREES):
    
    assert G.number_of_nodes() > 0
    assert is_tree(G)

    print plant
   
    input_points = G.graph['input points']
    point_graph = G.subgraph(input_points)
    print point_graph.number_of_nodes(), 'points'
   
# ---------------------------------------
    tree_costs_fname = '%s/tree_costs.csv' % fronts_dir

    models_fname = '%s/models_%s.csv' % (output_dir, plant)
    output_fname = '%s/pareto_steiner_%s.csv' %  (output_dir, plant)
    tradeoff_fname = '%s/tradeoff_%s.csv' % (output_dir, plant)



# ---------------------------------------
    alphas, mcosts, scosts, first_time = pareto_front(G, point_graph,\
                                                      fronts_dir, figs_dir,\
                                                      viz_trees) 

    opt_mcost, opt_scost = min(mcosts), min(scosts)
# --------------------------------------- 
    plant_mcost, plant_scost = graph_costs(G, relevant_nodes=point_graph.nodes())
    
    plant_dist, plant_index = DIST_FUNC(mcosts, scosts, plant_mcost,\
                                        plant_scost)
    plant_closem = mcosts[plant_index] 
    plant_closes = scosts[plant_index]
    plant_alpha = alphas[plant_index]

# ---------------------------------------
    tradeoff = tradeoff_ratio(plant_mcost, opt_mcost, plant_scost, opt_scost)
    
# ---------------------------------------
    centroid_tree = centroid_mst(point_graph)

    centroid_mcost, centroid_scost = graph_costs(centroid_tree, relevant_nodes=point_graph.nodes())
    
    centroid_dist, centroid_index = DIST_FUNC(mcosts, scosts,\
                                              centroid_mcost,\
                                              centroid_scost)
    centroid_closem = mcosts[centroid_index]
    centroid_closes = scosts[centroid_index]
    centroid_alpha = alphas[centroid_index]


# ---------------------------------------
    centroid_success = int(centroid_dist <= plant_dist)
    centroid_ratio = centroid_dist / plant_dist
    if first_time:
        with open(models_fname, 'a') as models_file:
            models_file.write('%s, %s, %f\n' % (plant, 'plant', plant_dist))
            models_file.write('%s, %s, %f, %d, %f\n' % (plant,\
                                                            'centroid',\
                                                            centroid_dist,\
                                                            centroid_success,\
                                                            centroid_ratio))

# ---------------------------------------
    if first_time:
        with open(tree_costs_fname, 'w') as tree_costs_file:
            tree_costs_file.write('tree, mcost, scost\n')
            tree_costs_file.write('%s, %f, %f\n' % ('plant', plant_mcost,\
                                                    plant_scost))
            tree_costs_file.write('%s, %f, %f\n' % ('centroid', centroid_mcost,\
                                                    centroid_scost))
# ---------------------------------------
    #point_graph = complete_graph(point_graph)
    random_trials = 20

    for i in xrange(random_trials):
        rand_mst = random_mst(point_graph, euclidean=True)
        rand_mcost, rand_scost = graph_costs(rand_mst)
        rand_dist, rand_index = DIST_FUNC(mcosts, scosts, rand_mcost,\
                                          rand_scost)
        rand_success = int(rand_dist <= plant_dist)
        rand_ratio = rand_dist / plant_dist

        barabasi_mst = barabasi_tree(point_graph)
        barabasi_mcost, barabasi_scost = graph_costs(barabasi_mst)
        barabasi_dist, barabasi_index = DIST_FUNC(mcosts, scosts,\
                                                  barabasi_mcost,\
                                                  barabasi_scost)
        barabasi_success = int(barabasi_dist <= plant_dist)
        barabasi_ratio = barabasi_dist / plant_dist

        with open(tree_costs_fname, 'a') as tree_costs_file:
            tree_costs_file.write('%s, %f, %f\n' % ('random', rand_mcost,\
                                                     rand_scost))
            tree_costs_file.write('%s, %f, %f\n' % ('barabasi',\
                                                     barabasi_mcost,\
                                                     barabasi_scost))

        with open(models_fname, 'a') as models_file:
            models_file.write('%s, %s, %f, %d, %f\n' % (plant, 'random',\
                                                        rand_dist,\
                                                        rand_success,\
                                                        rand_ratio))
            
            models_file.write('%s, %s, %f, %d, %f\n' % (plant, 'barabasi',\
                                                        barabasi_dist,\
                                                        barabasi_success,\
                                                        barabasi_ratio))

# ---------------------------------------
    def remove_spaces(string):
        return string.replace(' ', '')

    def remove_commas(string):
        return string.replace(',', '')
     
    if output and first_time:
        write_items = [plant, point_graph.number_of_nodes()]
        
        write_items.append(plant_alpha)
        
        write_items = map(str, write_items)
        write_items = map(remove_commas, write_items)
        write_items = ', '.join(write_items)
        with open(output_fname, 'a') as output_file:
            output_file.write('%s\n' % write_items)
        
        with open(tradeoff_fname, 'a') as tradeoff_file:
            tradeoff_file.write('%s, %f\n' % (plant, tradeoff))

def pareto_analysis_plants(names=None, viz_trees=VIZ_TREES, plot=False):
    datasets_dir = DATASETS_DIR
    for plant_file in os.listdir(datasets_dir):
        plant = plant_file[:-7]
        if names != None and plant not in names:
            continue

        G = read_tree('%s/%s' % (datasets_dir, plant_file))
         
        fronts_dir = '%s/%s' % (FRONTS_DIR, plant)
        fronts_dir = fronts_dir.replace(' ', '_')

        output_dir = TEMP_DIR
        output_dir = output_dir.replace(' ', '_')

        figs_dir = '%s/%s' % (FIGS_DIR, plant)

        figs_dir = figs_dir.replace(' ', '_')
       
        os.system('mkdir -p %s' % fronts_dir)
        os.system('mkdir -p %s' % output_dir)
        if plot or viz_trees:
            os.system('mkdir -p %s' % figs_dir)

        pareto_analysis(G, plant, fronts_dir=fronts_dir, output_dir=output_dir,\
                        figs_dir=figs_dir, viz_trees=viz_trees)
        if plot:
            pareto_plot(fronts_dir, figs_dir, plant)
            pareto_plot(fronts_dir, figs_dir, log_plot=True, plant=plant)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-na', '--names', nargs='+', default=None)
    parser.add_argument('-v', '--viz_trees', action='store_true')
    parser.add_argument('-p', '--plot', action='store_true')

    args = parser.parse_args()
    names = args.names
    viz_trees = args.viz_trees
    plot = args.plot
    
    pareto_analysis_plants(names, viz_trees, plot)

if __name__ == '__main__':
    main()
