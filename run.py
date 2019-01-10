from __future__ import print_function

import matplotlib; matplotlib.use('Agg')
import argparse
import cleanlog
import random
import pickle
import sys
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dohlee.plot as plot

from queue import Queue
from numba import jit
from multiprocessing import Pool
from collections import defaultdict
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from dohlee.thread import threaded
from sklearn.preprocessing import normalize

CONV_THRESHOLD = 0.000001

logger = cleanlog.ColoredLogger('NP4')
plot.set_style()

### Debug options. Comment it out if unneeded.
logger.setLevel(cleanlog.DEBUG)

# Fix random seed for reproducibility.
np.random.seed(419)

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--network', required=True, help='Input network file in edge list format.')
    parser.add_argument('-o', '--output', required=True, help='Output prefix.')
    parser.add_argument('-s', '--seed', required=True, help='Seed list file.')
    parser.add_argument('-e', '--restart_prob', type=float, default=0.7,
                        help='Restart probability for random walk. (default 0.7)')
    parser.add_argument('-w', '--weighted', action='store_true', default=False)
    parser.add_argument('-f', '--sf', type=float, default=20.0,
                        help='SF cutoff (default 20).')
    parser.add_argument('-x', '--iterations', type=int, default=1000,
                        help='Number of iterations within each of the procedure. (default 1000)')
    parser.add_argument('-k', '--sampled-seeds', type=int, default=100,
                        help='Number of sampled seeds for each of the iterations. (default 100)')
    parser.add_argument('-t', '--threads', type=int, default=1,
                        help='Number of threads to use. (default 1)')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='Increase verbosity.')
    return parser.parse_args()

@jit
def fit_gaussian(values):
    """Fits gaussian to 1-d values and returns the mean, and variance of the fitted function."""
    values = np.array(values).reshape(len(values), 1)
    gmm = GaussianMixture()
    gmm.fit(values)

    return gmm.means_[0, 0], gmm.covariances_[0, 0, 0]

@jit
def setup_p0(num_nodes, seed_indices, num_sampled_seeds, num_iterations):
    """Setup initial probability vector by sampling `num_sampled_seeds` seeds from seed_indices."""
    p0 = np.zeros([num_nodes, num_iterations])
    for i in range(num_iterations):
        p0[np.random.choice(a=seed_indices, size=num_sampled_seeds, replace=False), i] = 1. / num_sampled_seeds;

    return p0

@jit
def run_network_propagation(network, num_nodes, seed_indices, num_sampled_seeds, num_iterations, restart_prob):
    """
    Run network propagation by seed sampling. For each of `num_iterations` iterations,
    the number of sampled seeds will be `num_sampled_seeds`.

    :returns numpy array: A numpy array containing rank-mean information of all genes.
    """
    logger.debug('Running network propagation with %d iterations.' % num_iterations)
    p_0 = setup_p0(num_nodes, seed_indices, num_sampled_seeds, num_iterations)
    l1_norm_diff = 1

    p_curr = np.copy(p_0)
    # NOTE: This obligates the iterations to terminate only if all the p_vectors have been converged,
    # which means that even though some of the p_vectors are already converged, they can undergo some
    # Further iterations.
    iter_count = 0
    while np.any(l1_norm_diff > CONV_THRESHOLD):
        iter_count += 1
        p_next = (matrix @ p_curr) * (1 - restart_prob) + p_0 * restart_prob
        l1_norm_diff = np.linalg.norm(p_next - p_curr, ord=1, axis=0)
        p_curr = p_next

    # Compute rank means of each iteration.
    orders = np.argsort(-p_curr, axis=0)
    ranks = np.empty_like(orders)
    for i_col in range(num_iterations):
        # This makes columnwise-ranked rank matrix!
        ranks[orders[:, i_col], i_col] = np.arange(1, num_nodes + 1)
    # Compute average rank of each gene along all iterations.
    rank_means = np.mean(ranks, axis=1)

    logger.debug('Done on %d iterations.' % iter_count)

    return rank_means

def foreground_procedure(network, num_nodes, seed_indices, num_sampled_seeds, num_iterations, restart_prob):
    """
    Run foreground procedure for given seed indices.
    """
    return run_network_propagation(network, num_nodes, seed_indices, num_sampled_seeds, num_iterations, restart_prob)

def background_procedure(network, num_nodes, num_sampled_seeds, num_iterations, restart_prob):
    """
    Run background procedure for seed indices of the whole graph.
    """
    seed_indices = np.arange(num_nodes)
    return run_network_propagation(network, num_nodes, seed_indices, num_sampled_seeds, num_iterations, restart_prob)

def compute_seed_factor(f_rank_mean, b_rank_mean, seed_indices):
    fb_ratios = np.log2(f_rank_mean / b_rank_mean)

    seed_mask = np.zeros(len(f_rank_mean), dtype=bool)
    for i in seed_indices:
        seed_mask[i] = True

    seed_fb_ratios = fb_ratios[seed_mask]
    nonseed_fb_ratios = fb_ratios[~seed_mask]
    logger.debug('%d seeds and %d nonseeds.' % (len(seed_fb_ratios), len(nonseed_fb_ratios)))

    seed_mean, seed_var = fit_gaussian(seed_fb_ratios)
    nonseed_mean, nonseed_var = fit_gaussian(nonseed_fb_ratios)

    p_given_seed = norm.pdf(fb_ratios, loc=seed_mean, scale=np.sqrt(seed_var))
    p_given_nonseed = norm.pdf(fb_ratios, loc=nonseed_mean, scale=np.sqrt(nonseed_var))
    seed_factors = p_given_seed / p_given_nonseed

    return seed_factors

args = parse_argument()
if args.verbose:
    logger.setLevel(cleanlog.DEBUG)

# Instantiate input network.
if args.weighted:
    network = nx.read_weighted_edgelist(args.network)
else:
    network = nx.read_edgelist(args.network)

nodes = list(network.nodes())
num_nodes = len(nodes)

# Normalized adjacency matrix for network propagation iteration.
matrix = normalize(nx.to_numpy_matrix(network, dtype=np.float32), norm='l1', axis=0)

# Prepare seeds for foreground procedure.
seed_list = [l.strip() for l in open(args.seed).readlines()]
seed_indices = [nodes.index(s) for s in seed_list]

logger.debug('[Background procedure]')
background_rank_means = background_procedure(network, num_nodes, args.sampled_seeds, args.iterations, args.restart_prob)

logger.debug('[Foreground procedure]')
foreground_rank_means = foreground_procedure(network, num_nodes, seed_indices, args.sampled_seeds, args.iterations, args.restart_prob)

logger.debug('Computing seed factors.')
seed_factors = compute_seed_factor(foreground_rank_means, background_rank_means, seed_indices)

enhanced_seed_indices = np.where(seed_factors > args.sf)[0]
logger.debug('Found %d enhanced seeds from %d seeds.' % (len(enhanced_seed_indices), len(seed_indices)))

with open('%s_foreground_gene_rank_means.pkl' % args.output, 'wb') as f:
    pickle.dump(foreground_rank_means, f)

with open('%s_background_rank_means.pkl' % args.output, 'wb') as f:
    pickle.dump(background_rank_means, f)

with open('%s_seed.list' % args.output, 'w') as f:
    enhanced_seed_list = np.array(nodes)[enhanced_seed_indices] 
    print('\n'.join(enhanced_seed_list), file=f)
