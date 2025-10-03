import os
import sys
import time
import warnings
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)



if __name__ == '__main__':

    from script.config import args
    from script.utils.data_utils import load_data
    from script.utils.curv_utils import visualize_histogram_distribution
    from script.manifolds.pseudohyperboloid_sr import PseudoHyperboloidSR
    from script.manifolds.pseudohyperboloid import PseudoHyperboloid
    from script.manifolds.hyperboloid import Hyperboloid

    warnings.filterwarnings("ignore")
    model = 'QGT'
    dataset = 'cora'
    time_dim = 14
    space_dim = 2
    beta = torch.tensor(-1)
    feature_path = "output/lp/{}/{}/1234/embeddings.npy".format(model, dataset)
    output_path = "output/ablation/{}_embeddings.npy".format(model)

    print("Loading embedding")
    embedding = torch.from_numpy(np.load(feature_path)) # Q^{p, q}
    print(embedding.shape)
    manifold = PseudoHyperboloid(time_dim, space_dim)
    sub_manifold = Hyperboloid()
    embedding_sh = manifold.q_to_sh(manifold.extrinsic_map(embedding), beta, time_dim + 1)
    print(embedding_sh.shape)
    embedding_h = embedding_sh[:, time_dim+1:]
    assert not torch.isnan(embedding_h).any()
    embedding_h_pe = sub_manifold.to_poincare(embedding_h, beta.abs())
    assert not torch.isnan(embedding_h_pe).any()
    print(embedding_h_pe.shape)
    np.save(output_path, embedding_h_pe.detach().numpy())

    
