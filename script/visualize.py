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
    from script.utils.curv_utils import visualize_histogram_distribution

    warnings.filterwarnings("ignore")
    args.datapath = "../data/node_classification/{}/".format(args.dataset)

    data = load_data(args)
    t0 = time.time()
    print("Dataset: ", args.dataset)
    print("Number of nodes: ", len(data['node_list']))
    src, dst = data['edge_index']
    matches = src != dst
    data['edge_index'] = data['edge_index'][:, matches]
    print("Number of edges: ", data['edge_index'].shape[-1])

    data_sc = np.load("./output/{}/gsc.npy".format(args.dataset))

    print("Visualizing input GSC distribution...")
    filepath2 = "./output/{}/{}_input_gsc_distribution.pdf".format(args.dataset, args.dataset)
    visualize_histogram_distribution(args.dataset, data_sc, filepath2)
    print("Saved!")
