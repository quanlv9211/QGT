import os
import sys
import time
import warnings
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)



if __name__ == '__main__':

    from script.config import args
    from script.utils.data_utils import load_data
    from script.utils.curv_utils import sectional_curvature_distribution, find_ideal_hyperboloid, visualize_gsc_distribution, visualize_histogram_distribution, visualize_ideal_input_distribution

    warnings.filterwarnings("ignore")
    args.datapath = "../data/node_classification/{}/".format(args.dataset)
    data = load_data(args)
    t0 = time.time()
    print("Dataset: ", args.dataset)
    print("Number of nodes: ", len(data['node_list']))
    src, dst = data['edge_index']
    matches = src != dst
    data['edge_index'] = data['edge_index'][:, matches]
    print("Number of edges: ", data['edge_index'].shape[1])

    data_sc, data_mean, data_std = sectional_curvature_distribution(args, data['edge_index'], data['node_list'])
    # print("Dataset: ", args.dataset)
    # data_sc = np.load("./output/{}/gsc.npy".format(args.dataset))
    # data_mean = np.mean(data_sc)
    # data_std = np.std(data_sc)
    print(data_mean)
    print(data_std)

    dim = 17
    q, p = find_ideal_hyperboloid(dim, data_mean, data_std)

    print("Time: {:.3f}".format(time.time()-t0))

    epsilon = 0.1
    mean = []
    std = []
    p_i = []
    n = int((dim - 1) / 2)
    dim = int(dim)
    std_max = 5 * abs(data_mean)
    std_min = 0.5 * abs(data_mean)

    if data_mean > epsilon:
        print("Spherical")
        for i in range(0, n):
            std_i = (std_min + (std_max - std_min) * ((2 * i) / (dim - 3)))
            mean.append(abs(data_mean))
            std.append(std_i)
            p_i.append(i)
    elif data_mean < -epsilon:
        print("Hyperbolic")
        for i in range(n+1, dim):
            std_i = (std_max - (std_max - std_min) * ((2 * i - dim - 1) / (dim - 3)))
            mean.append(-abs(data_mean))
            std.append(std_i)
            p_i.append(i)
    else:
        print("Mixed")
        mean.append(0)
        std.append(1)
        p_i.append(n)

    print("Visualizing ideal GSC distributions...")
    filepath1 = "./output/{}/".format(args.dataset)
    if not os.path.isdir(filepath1):
       os.makedirs(filepath1)

    visualize_gsc_distribution(mean, std, data_sc, p_i, filepath1 + "all_ideal_gsc_distributions.pdf")
    print("Saved!")

    # plot the histogram and Gaussian of dataset, save fig
    print("Visualizing input GSC distribution...")
    filepath2 = "./output/{}/input_gsc_distribution.pdf".format(args.dataset)
    visualize_histogram_distribution(args.dataset, data_sc, filepath2)
    print("Saved!")

    # save the numpy array of curvatures
    filepath3 = "./output/{}/gsc.npy".format(args.dataset)
    np.save(filepath3, data_sc)
    print("Saved gsc numpy!")

    print("Visualizing input and ideal GSC distributions...")
    filepath4 = "./output/{}/input_ideal_gsc_distribution.pdf".format(args.dataset)
    visualize_ideal_input_distribution(mean, std, data_sc, data_mean, data_std, p_i, filepath4)
    print("Saved!")