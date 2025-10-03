"""Curvature Distribution"""

import numpy as np
import networkx as nx
import math
import matplotlib.pyplot as plt
from scipy.stats import norm

def kl_gaussian(mu1, sigma1, mu2, sigma2):
    return np.log(sigma2 / sigma1) + (sigma1**2 + (mu1 - mu2)**2) / (2 * sigma2**2) - 0.5

def sectional_curvature_distribution(args, edge_index, node_list):
    # A graph G is represented by an edge index and a list of nodes
    # First we sample the list A
    edge_index = edge_index.numpy() # the edge_index must be numpy array
    print("Sampling a list of node...")
    number_sample = args.number_sample_A_set
    sampled_nodes = np.random.choice(node_list, number_sample, replace=False)
    print("Done")
    G = nx.Graph()
    G.add_edges_from(edge_index.T)

    print("Precomputing shortest paths...")
    shortest_paths_dict = {}
    for node in sampled_nodes:
        if G.has_node(node):
            shortest_paths_dict[node] = nx.single_source_shortest_path_length(G, node)
    print("Done")

    print("Calculating sectional curvatures...")
    sectional_curvature_list = []
    n_s = args.number_sample_neighbors
    for m in node_list:
        if not G.has_node(m):
            continue
        curv_m = 0
        cnt_ = 0
        neighbors = np.array(list(G.neighbors(m)))
        if len(neighbors) < 2:
            continue
        n = len(neighbors)
        n_ss = min(n * (n - 1), n_s) #avoid duplicate
        for i in range(n_ss):
            sampled_neighbors = np.random.choice(neighbors, 2, replace=False)
            b = sampled_neighbors[0]
            c = sampled_neighbors[1]
            if G.has_edge(b, c):
                d_bc = 1
            else:
                d_bc = 2
            curv_mbc = 0
            cnt = 0
            for a in shortest_paths_dict:
                # print(a, m)
                # if a in shortest_paths_dict:
                d_am = shortest_paths_dict[a].get(m, float('inf'))
                if math.isinf(d_am): # a and m are disconnected
                    continue
                if d_am == 0: # a == m
                    continue
                cnt += 1
                d_ab = shortest_paths_dict[a].get(b, float('inf'))
                d_ac = shortest_paths_dict[a].get(c, float('inf'))
                curv_mbca = ((d_am / 2) + (d_bc**2 / (8 * d_am)) - ((d_ab**2 + d_ac**2) / (4 * d_am)))
                assert not math.isnan(curv_mbca)
                curv_mbc += curv_mbca

            if cnt == 0:
                continue
            
            curv_mbc /= cnt
            curv_m += curv_mbc
            cnt_ += 1
        if cnt_ == 0:
            continue
        curv_m /= cnt_
        sectional_curvature_list.append(curv_m)
    print("Done")

    print("Calculating mean and variance...")
    sectional_curvature_np = np.array(sectional_curvature_list)
    mean = np.mean(sectional_curvature_np)
    std = np.std(sectional_curvature_np)
    print("Done")

    print("Sectional curvatures done")
    return sectional_curvature_np, mean, std


def find_ideal_hyperboloid(dim, true_mean, true_std):
    # (dim - 1) should be even
    epsilon = 0.1
    dim = int(dim)
    n = int((dim - 1) / 2)
    std_max = 5 * abs(true_mean)
    std_min = 0.5 * abs(true_mean)
    if (true_mean <= epsilon) and (true_mean >= -epsilon): # uniform
        p_i = n
        print("Dim: ", n, " KL: ", 0)
    elif true_mean > epsilon: # spherical
        p_i = 0
        min_kl = 10000000
        for i in range(0, n):
            std_i = (std_min + (std_max - std_min) * ((2 * i) / (dim - 3)))
            kl = kl_gaussian(true_mean, true_std, abs(true_mean), std_i)
            print("Dim: ", i, " KL: ", kl)
            if kl < min_kl:
                min_kl = kl
                p_i = i

    else: # hyperbolic
        p_i = 0
        min_kl = 10000000
        for i in range(n+1, dim):
            std_i = (std_max - (std_max - std_min) * ((2 * i - dim - 1) / (dim - 3)))
            kl = kl_gaussian(true_mean, true_std, -abs(true_mean), std_i)
            print("Dim: ", i, " KL: ", kl)
            if kl < min_kl:
                min_kl = kl
                p_i = i
    q_i = dim - 1 - p_i
    print("p:", p_i)
    print("q:", q_i)
    return q_i, p_i

#def visualize_all_gsc_distribution(true_mean, dim, filepath=None):
#    # visualize different ideal GSC
#    # (dim - 1) should be even
#    mean = []
#    std = []
#    p_i = []
#    n = (dim - 1) / 2
#    std_max = 5 * abs(true_mean)
#    std_min = 0.5 * abs(true_mean)

#    for i in range(0, n):
#        std_i = (std_min + (std_max - std_min) * ((2 * i) / (dim - 3)))
#        mean.append(abs(true_mean))
#        std.append(std_i)
#        p_i.append(i)

#    mean.append(0)
#    std.append(1)
#    p_i.append(n)

#    for i in range(n+1, dim):
#        std_i = (std_max - (std_max - std_min) * ((2 * i - dim - 1) / (dim - 3)))
#        mean.append(-abs(true_mean))
#        std.append(std_i)
#        p_i.append(i)

#    # Define the x range for plotting
#    x = np.linspace(-5, 5, 1000)

#    # Create the plot
#    fig = plt.figure(figsize=(8, 5))

#    # Plot each Gaussian distribution
#    for mu, sigma, p in zip(mean, std, p_i):
#        y = norm.pdf(x, mu, sigma)  # Compute the Gaussian PDF
#        plt.plot(x, y, label=f"$p_i$={p}")

#    # Customize plot
#    plt.xlabel("x")
#    plt.ylabel("Probability Density")
#    plt.title("Gaussian Distributions")
#    plt.legend()
#    plt.grid()

#    # Save the plot
#    if filepath is None:
#        plt.savefig("all_gsc_distributions.pdf", dpi=fig.dpi, bbox_inches="tight")
#    else:
#        plt.savefig(filepath, dpi=fig.dpi, bbox_inches="tight")

def visualize_gsc_distribution(list_mean, list_std, list_curv, list_p=None, filepath=None):
    # visualize different ideal GSC
    # (dim - 1) should be even
 

    # Define the x range for plotting
    x = np.linspace(min(list_curv), max(list_curv), 1000)

    # Create the plot
    fig = plt.figure(figsize=(8, 5))

    if list_p is not None:
        # Plot each Gaussian distribution
        for mu, sigma, p in zip(list_mean, list_std, list_p):
            y = norm.pdf(x, mu, sigma)  # Compute the Gaussian PDF
            plt.plot(x, y, label=f"$p_i$={p}")
    else:
        # Plot each Gaussian distribution
        for mu, sigma in zip(list_mean, list_std):
            y = norm.pdf(x, mu, sigma)  # Compute the Gaussian PDF
            plt.plot(x, y, label=f"$\mu$={mu}, $\sigma$={sigma}")

    # Customize plot
    plt.xlabel("x")
    plt.ylabel("Probability Density")
    plt.title("Gaussian Distributions")
    plt.legend()
    plt.grid()

    # Save the plot
    if filepath is None:
        plt.savefig("gsc_distribution.pdf", dpi=fig.dpi, bbox_inches="tight")
    else:
        plt.savefig(filepath, dpi=fig.dpi, bbox_inches="tight")

def visualize_histogram_distribution(dataset, list_curv, filepath=None):
    # Calculate mean and std of the data 
    mu = np.mean(list_curv)
    sigma = np.std(list_curv)

    plt.rcParams.update({'font.size': 14})
    fig = plt.figure(figsize=(5, 5))    

    plt.hist(list_curv, bins=30, density=True, color='skyblue', edgecolor='black')

    x = np.linspace(min(list_curv), max(list_curv), 1000)
    pdf = norm.pdf(x, mu, sigma)
    plt.plot(x, pdf, 'r-', linewidth=2)

    plt.axvline(mu, color='red', linestyle='--', label='mean')

    plt.xlabel("Sectional curvature" , fontsize = 14)
    plt.ylabel("Density" , fontsize = 14)
    plt.title(dataset , fontsize = 14)
    plt.legend()
    #plt.grid()

    if filepath is None:
        plt.savefig("gsc_distribution.pdf", dpi=fig.dpi, bbox_inches="tight")
    else:
        plt.savefig(filepath, dpi=fig.dpi, bbox_inches="tight")


def visualize_ideal_input_distribution(list_mean, list_std, list_curv, true_mean, true_curve, list_p=None, filepath=None):
    # Define the x range for plotting
    x = np.linspace(min(list_curv), max(list_curv), 1000)

    # Create the plot
    fig = plt.figure(figsize=(8, 5))

    if list_p is not None:
        # Plot each Gaussian distribution
        for mu, sigma, p in zip(list_mean, list_std, list_p):
            y = norm.pdf(x, mu, sigma)  # Compute the Gaussian PDF
            plt.plot(x, y, label=f"$p_i$={p}")
    else:
        # Plot each Gaussian distribution
        for mu, sigma in zip(list_mean, list_std):
            y = norm.pdf(x, mu, sigma)  # Compute the Gaussian PDF
            plt.plot(x, y, label=f"$\mu$={mu}, $\sigma$={sigma}")

    y = norm.pdf(x, true_mean, true_curve)
    plt.plot(x, y, color='black', label=f"input")

    # Customize plot
    plt.xlabel("x")
    plt.ylabel("Probability Density")
    plt.title("Gaussian Distributions")
    plt.legend()
    plt.grid()

    # Save the plot
    if filepath is None:
        plt.savefig("gsc_distribution.pdf", dpi=fig.dpi, bbox_inches="tight")
    else:
        plt.savefig(filepath, dpi=fig.dpi, bbox_inches="tight")