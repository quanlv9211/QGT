"""Data utils functions for pre-processing and data loading."""
import os
import pickle as pkl
import sys
import json

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from .preprocessing import *
import random
import pandas as pd

import torch.nn as nn
from torch_geometric.utils import to_undirected
from ogb.nodeproppred import NodePropPredDataset
from ogb.linkproppred import PygLinkPropPredDataset
import scipy
from sklearn.preprocessing import label_binarize
from tqdm.auto import tqdm

from torch_sparse import SparseTensor

def load_data(args, logger=None):
    if args.task == 'nc':
        data = load_data_nc(args.dataset, args.use_feats, args.datapath, args.split_seed)
    elif args.task == 'lp':
        data = load_data_lp(args.dataset, args.use_feats, args.datapath, args.split_seed)
    else:
        raise Exception('This code does not do this task')

    data['features'] = process_features(data['features'], args.normalize_feats)

    return data


# ############### FEATURES PROCESSING ####################################

def process(adj, features, normalize_adj, normalize_feats):
    if sp.isspmatrix(features):
        features = np.array(features.todense())
    if normalize_feats:
        features = normalize(features)
    features = torch.Tensor(features)
    if normalize_adj:
        adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, features

def process_features(features, normalize_feats=False):
    if sp.isspmatrix(features):
        features = np.array(features.todense())
    if normalize_feats:
        features = normalize(features)
    features = torch.Tensor(features)
    return features


def normalize(mx):
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def augment(adj, features, normalize_feats=True):
    deg = np.squeeze(np.sum(adj, axis=0).astype(int))
    deg[deg > 5] = 5
    deg_onehot = torch.tensor(np.eye(6)[deg], dtype=torch.float).squeeze()
    const_f = torch.ones(features.size(0), 1)
    features = torch.cat((features, deg_onehot, const_f), dim=1)
    return features


# ############### DATA SPLITS #####################################################


def mask_edges(adj, val_prop, test_prop, seed):
    np.random.seed(seed)  # get tp edges
    x, y = sp.triu(adj).nonzero()
    pos_edges = np.array(list(zip(x, y)))
    np.random.shuffle(pos_edges)
    # get tn edges
    x, y = sp.triu(sp.csr_matrix(1. - adj.toarray())).nonzero()
    neg_edges = np.array(list(zip(x, y)))
    np.random.shuffle(neg_edges)

    m_pos = len(pos_edges)
    n_val = int(m_pos * val_prop)
    n_test = int(m_pos * test_prop)
    val_edges, test_edges, train_edges = pos_edges[:n_val], pos_edges[n_val:n_test + n_val], pos_edges[n_test + n_val:]
    val_edges_false, test_edges_false = neg_edges[:n_val], neg_edges[n_val:n_test + n_val]
    train_edges_false = np.concatenate([neg_edges, val_edges, test_edges], axis=0)
    adj_train = sp.csr_matrix((np.ones(train_edges.shape[0]), (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T
    return adj_train, torch.LongTensor(train_edges), torch.LongTensor(train_edges_false), torch.LongTensor(val_edges), \
           torch.LongTensor(val_edges_false), torch.LongTensor(test_edges), torch.LongTensor(
            test_edges_false)  


def split_data(labels, val_prop, test_prop, seed):
    np.random.seed(seed)
    nb_nodes = labels.shape[0]
    all_idx = np.arange(nb_nodes)
    pos_idx = labels.nonzero()[0]
    neg_idx = (1. - labels).nonzero()[0]
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    pos_idx = pos_idx.tolist()
    neg_idx = neg_idx.tolist()
    nb_pos_neg = min(len(pos_idx), len(neg_idx))
    nb_val = round(val_prop * nb_pos_neg)
    nb_test = round(test_prop * nb_pos_neg)
    idx_val_pos, idx_test_pos, idx_train_pos = pos_idx[:nb_val], pos_idx[nb_val:nb_val + nb_test], pos_idx[
                                                                                                   nb_val + nb_test:]
    idx_val_neg, idx_test_neg, idx_train_neg = neg_idx[:nb_val], neg_idx[nb_val:nb_val + nb_test], neg_idx[
                                                                                                   nb_val + nb_test:]
    return idx_val_pos + idx_val_neg, idx_test_pos + idx_test_neg, idx_train_pos + idx_train_neg


def bin_feat(feat, bins):
    digitized = np.digitize(feat, bins)
    return digitized - digitized.min()


# ############### NODE CLASSIFICATION DATA LOADERS ####################################

def load_data_nc(dataset, use_feats, data_path, split_seed):
    if dataset in ['cora', 'citeseer', 'pubmed']:
        edge_index, node_list, features, labels, idx_train, idx_val, idx_test = load_citation_data(
            dataset, use_feats, data_path, split_seed
        )
    elif dataset in ['airport']:
        edge_index, node_list, features, labels, idx_train, idx_val, idx_test = load_airport_data(dataset, data_path, split_seed)
    elif dataset in ['disease_nc']:
        edge_index, node_list, features, labels, idx_train, idx_val, idx_test = load_disease_data(dataset, data_path, split_seed)
    elif dataset in ['ogbn-products', 'ogbn-papers100M', 'ogbn-arxiv', 'ogbn-proteins']:
        edge_index, node_list, features, labels, idx_train, idx_val, idx_test = load_ogb_data(dataset, data_path)
    elif dataset in ['fb100']:
        edge_index, node_list, features, labels, idx_train, idx_val, idx_test = load_fb100_dataset(dataset, data_path)
    elif dataset in ['twitch_gamer']:
        edge_index, node_list, features, labels, idx_train, idx_val, idx_test = load_twitch_gamer_dataset(dataset, data_path)
    else:
        raise Exception('pls provide the dataset')
    labels = torch.LongTensor(labels)
    data = {'edge_index': edge_index, 'node_list': node_list, 'features': features, 'labels': labels, 'idx_train': idx_train, 'idx_val': idx_val, 'idx_test': idx_test}
    return data


# ############### DATASETS ####################################


def load_citation_data(dataset_str, use_feats, data_path, split_seed=None):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(os.path.join(data_path, "ind.{}.{}".format(dataset_str, names[i])), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(os.path.join(data_path, "ind.{}.test.index".format(dataset_str)))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
       # Fix citeseer dataset (there are some isolated nodes in the graph)
       # Find isolated nodes, add them as zero-vecs into the right position
       test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
       tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
       tx_extended[test_idx_range - min(test_idx_range), :] = tx
       tx = tx_extended
       ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
       ty_extended[test_idx_range - min(test_idx_range), :] = ty
       ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = np.argmax(labels, 1)

    idx_test = test_idx_range.tolist()
    idx_train = list(range(len(y)))
    idx_val = range(len(y), len(y) + 500)

    G = nx.from_dict_of_lists(graph)
    edge_index = np.array(list(G.edges)).T # (2 , E)
    edge_index = torch.from_numpy(edge_index).to(torch.int)
    edge_index = to_undirected(edge_index)
    node_list = np.array(list(G.nodes))

    #adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    if not use_feats:
        features = sp.eye(len(node_list))
        rand_feature = np.random.uniform(low=-0.01, high=0.01, size=(len(node_list) ,features.shape[1]))
        features = features + sp.csr_matrix(rand_feature)
    return edge_index, node_list, features, labels, idx_train, idx_val, idx_test


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_json_data(dataset_str, use_feats, data_path):
    object_to_idx = {}
    idx_counter = 0
    edges = []
    with open(os.path.join(data_path, "{}.edges.csv".format(dataset_str)), 'r') as f:
        all_edges = f.readlines()
    for line in all_edges:
        n1, n2 = line.rstrip().split(',')
        if n1 in object_to_idx:
            i = object_to_idx[n1]
        else:
            i = idx_counter
            object_to_idx[n1] = i
            idx_counter += 1
        if n2 in object_to_idx:
            j = object_to_idx[n2]
        else:
            j = idx_counter
            object_to_idx[n2] = j
            idx_counter += 1
        edges.append((i, j))
    adj = np.zeros((len(object_to_idx), len(object_to_idx)))
    for i, j in edges:
        adj[i, j] = 1.  # comment this line for directed adjacency matrix
        adj[j, i] = 1.
    if use_feats:
        features = sp.load_npz(os.path.join(data_path, "{}.feats.npz".format(dataset_str)))
    else:
        features = sp.eye(adj.shape[0])
    labels = np.load(os.path.join(data_path, "{}.labels.npy".format(dataset_str)))
    return sp.csr_matrix(adj), features, labels


def load_airport_data(dataset_str, data_path, seed):
    graph = pkl.load(open(os.path.join(data_path, dataset_str + '.p'), 'rb'))
    adj = nx.adjacency_matrix(graph)
    features = np.array([graph.nodes[u]['feat'] for u in graph.nodes()])
    label_idx = 4
    labels = features[:, label_idx]
    features = features[:, :label_idx]
    labels = bin_feat(labels, bins=[7.0/7, 8.0/7, 9.0/7])
    num_nodes = adj.shape[0]
    node_list = list(range(num_nodes))
    features = torch.tensor(features, dtype=torch.float)
    val_prop, test_prop = 0.15, 0.15
    idx_val, idx_test, idx_train = split_data(labels, val_prop, test_prop, seed)
    deg = np.squeeze(np.sum(adj, axis=0).astype(int))
    deg[deg > 5] = 5
    deg_onehot = torch.tensor(np.eye(6)[deg], dtype=torch.float).squeeze()
    const_f = torch.ones(features.size(0), 1)
    features = torch.cat((features, deg_onehot, const_f), dim=1)
    edge_index = torch.tensor(adj.nonzero(), dtype=torch.long)
    return edge_index, node_list, features, labels, idx_train, idx_val, idx_test


def load_disease_data(dataset_str, data_path, seed):
    object_to_idx = {}
    idx_counter = 0
    edges = []

    with open(os.path.join(data_path, dataset_str + ".edges.csv"), 'r') as f:
        all_edges = f.readlines()
    for line in all_edges:
        n1, n2 = line.rstrip().split(',')
        if n1 in object_to_idx:
            i = object_to_idx[n1]
        else:
            i = idx_counter
            object_to_idx[n1] = i
            idx_counter += 1
        if n2 in object_to_idx:
            j = object_to_idx[n2]
        else:
            j = idx_counter
            object_to_idx[n2] = j
            idx_counter += 1
        edges.append((i, j))
    adj = np.zeros((len(object_to_idx), len(object_to_idx)))
    for i, j in edges:
        #adj[i, j] = 1.  # comment this line for directed adjacency matrix
        adj[j, i] = 1.
    features = sp.load_npz(os.path.join(data_path, dataset_str + ".feats.npz"))
    if sp.issparse(features):
        features = features.toarray()
    features = normalize(features)
    features = torch.tensor(features, dtype=torch.float)

    labels = np.load(os.path.join(data_path, dataset_str + ".labels.npy"))
    val_prop, test_prop = 0.10, 0.60
    idx_val, idx_test, idx_train = split_data(labels, val_prop, test_prop, seed)
    num_nodes = adj.shape[0]
    node_list = list(range(num_nodes))
    edge_index = torch.tensor(adj.nonzero(), dtype=torch.long)

    return edge_index, node_list, features, labels, idx_train, idx_val, idx_test

def to_sparse_tensor(edge_index, edge_feat, num_nodes):
    """ converts the edge_index into SparseTensor
    """
    num_edges = edge_index.size(1)

    (row, col), N, E = edge_index, num_nodes, num_edges
    perm = (col * N + row).argsort()
    row, col = row[perm], col[perm]

    value = edge_feat[perm]
    adj_t = SparseTensor(row=col, col=row, value=value,
                         sparse_sizes=(N, N), is_sorted=True)

    # Pre-process some important attributes.
    adj_t.storage.rowptr()
    adj_t.storage.csr2csc()

    return adj_t

def load_ogb_data(dataset_str, data_path):
    if not os.path.isdir(data_path):
        os.makedirs(data_path)
    
    dataset = NodePropPredDataset(name=dataset_str, root=data_path)
    splitted = dataset.get_idx_split()
    idx_train = splitted['train']
    idx_val = splitted['valid']
    idx_test = splitted['test']
    graph, labels = dataset[0]
    features = torch.tensor(graph["node_feat"]).float().contiguous() if graph["node_feat"] is not None else None
    labels = labels.squeeze()
    edge_index = graph["edge_index"]
    edge_index = torch.from_numpy(edge_index).to(torch.int)
    if dataset_str == 'ogbn-proteins': # obg proteins does not have node features
        edge_index_ = to_sparse_tensor(edge_index,
                                   graph['edge_feat'], graph['num_nodes'])
        features = edge_index_.mean(dim=1)
    num_nodes = graph['num_nodes']
    node_list = list(range(num_nodes))

    return edge_index, node_list, features, labels, idx_train, idx_val, idx_test


def load_fb100(data_path, filename):
    # e.g. filename = Rutgers89 or Cornell5 or Wisconsin87 or Amherst41
    # columns are: student/faculty, gender, major,
    #              second major/minor, dorm/house, year/ high school
    # 0 denotes missing entry
    mat = scipy.io.loadmat(data_path  + filename + '.mat')
    A = mat['A']
    metadata = mat['local_info']
    return A, metadata

def load_fb100_dataset(dataset_str, data_path, subname='Penn94'):
    A, metadata = load_fb100(data_path, subname)
    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    metadata = metadata.astype(np.int32)
    labels = metadata[:, 1] - 1  # gender label, -1 means unlabeled
    #labels = torch.tensor(labels)

    # make features into one-hot encodings
    feature_vals = np.hstack(
        (np.expand_dims(metadata[:, 0], 1), metadata[:, 2:]))
    features = np.empty((A.shape[0], 0))
    for col in range(feature_vals.shape[1]):
        feat_col = feature_vals[:, col]
        feat_onehot = label_binarize(feat_col, classes=np.unique(feat_col))
        features = np.hstack((features, feat_onehot))

    features = torch.tensor(features, dtype=torch.float)
    num_nodes = metadata.shape[0]
    node_list = list(range(num_nodes))

    splits_lst = np.load(data_path + 'splits/' + "{}-{}-splits.npy".format(dataset_str, subname), allow_pickle=True)
    for i in range(len(splits_lst)):
        for key in splits_lst[i]:
            if not torch.is_tensor(splits_lst[i][key]):
                splits_lst[i][key] = torch.as_tensor(splits_lst[i][key])
    split_index = splits_lst[0]
    idx_train = split_index['train']
    idx_val = split_index['valid']
    idx_test = split_index['test']

    return edge_index, node_list, features, labels, idx_train, idx_val, idx_test

def load_twitch_gamer(nodes, task="dead_account"):
    nodes = nodes.drop('numeric_id', axis=1)
    nodes['created_at'] = nodes.created_at.replace('-', '', regex=True).astype(int)
    nodes['updated_at'] = nodes.updated_at.replace('-', '', regex=True).astype(int)
    one_hot = {k: v for v, k in enumerate(nodes['language'].unique())}
    lang_encoding = [one_hot[lang] for lang in nodes['language']]
    nodes['language'] = lang_encoding
    
    if task is not None:
        label = nodes[task].to_numpy()
        features = nodes.drop(task, axis=1).to_numpy()
    
    return label, features

def load_twitch_gamer_dataset(dataset_str, data_path, task="mature", normalize=True):
    
    edges = pd.read_csv(data_path + 'twitch_gamer_edges.csv')
    nodes = pd.read_csv(data_path + 'twitch_gamer_features.csv')
    edge_index = torch.tensor(edges.to_numpy()).t().type(torch.LongTensor)
    num_nodes = len(nodes)
    labels, features = load_twitch_gamer(nodes, task)
    node_feat = torch.tensor(features, dtype=torch.float)
    if normalize:
        node_feat = node_feat - node_feat.mean(dim=0, keepdim=True)
        node_feat = node_feat / node_feat.std(dim=0, keepdim=True)
    features = node_feat
    node_list = list(range(num_nodes))
    splits_lst = np.load(data_path + 'splits/' + "{}-splits.npy".format(dataset_str), allow_pickle=True)
    for i in range(len(splits_lst)):
        for key in splits_lst[i]:
            if not torch.is_tensor(splits_lst[i][key]):
                splits_lst[i][key] = torch.as_tensor(splits_lst[i][key])
    split_index = splits_lst[0]
    idx_train = split_index['train']
    idx_val = split_index['valid']
    idx_test = split_index['test']
    return edge_index, node_list, features, labels, idx_train, idx_val, idx_test

def is_tree(edge_index):
    G = nx.Graph()
    edges = list(zip(edge_index[0], edge_index[1]))
    G.add_edges_from(edges)
    num_nodes = edge_index.max() + 1
    G.add_nodes_from(range(num_nodes))
    return nx.is_tree(G)

def create_tree(b_factor, depth, max_nodes):
    num_nodes = max_nodes
    edge_matrix = np.zeros((num_nodes, num_nodes))
        
    for i in range(num_nodes-b_factor**depth):
        edge_matrix[i, b_factor*i+1:b_factor*(i+1)+1] = 1

    edge_matrix = edge_matrix + np.transpose(edge_matrix)
    rows, cols = np.nonzero(edge_matrix)
    edge_index1 = np.stack([rows, cols], axis=0)

    #print(is_tree(edge_index1))
    assert is_tree(edge_index1) == True
    existing_edges = set((u, v) for u, v in zip(edge_index1[0], edge_index1[1]))
    new_edges = set()
    num_nodes = max_nodes
    while len(new_edges) < 1000:
        u = random.randint(0, num_nodes - 1)
        v = random.randint(0, num_nodes - 1)
        if u == v:
            continue 
        if (u, v) in existing_edges or (u, v) in new_edges:
            continue 
        if (v, u) in existing_edges or (v, u) in new_edges:
            continue 
        new_edges.add((u, v))
        new_edges.add((v, u))
    new_edge_index = np.array(list(new_edges)).reshape((2, -1)) 
    edge_index2 = np.concatenate([edge_index1, new_edge_index], axis=1)

    existing_edges = set((u, v) for u, v in zip(edge_index2[0], edge_index2[1]))
    new_edges = set()
    while len(new_edges) < 1000:
        u = random.randint(0, num_nodes - 1)
        v = random.randint(0, num_nodes - 1)
        if u == v:
            continue 
        if (u, v) in existing_edges or (u, v) in new_edges:
            continue
        if (v, u) in existing_edges or (v, u) in new_edges:
            continue 
        new_edges.add((u, v))
        new_edges.add((v, u))
    new_edge_index = np.array(list(new_edges)).reshape((2, -1)) 
    edge_index3 = np.concatenate([edge_index2, new_edge_index], axis=1)

    return edge_index1, edge_index2, edge_index3

def create_tree_dataset(split_seed):
    # this code only needs to run one time.
    # create tree1 dataset before creating tree2 and tree3
    number_of_nodes = 5461
    depth = 6
    braching_factor = 4

    edge_index1, edge_index2, edge_index3 = create_tree(braching_factor, depth, number_of_nodes)
    edge_index1 = torch.from_numpy(edge_index1)
    edge_index2 = torch.from_numpy(edge_index2)
    edge_index3 = torch.from_numpy(edge_index3)
    node_list = list(range(number_of_nodes))
    features = torch.eye(number_of_nodes, dtype=torch.float)

    val_prop = 0.05
    test_prop = 0.1
    train_edges, train_edges_neg, val_edges, val_edges_neg, test_edges, test_edges_neg = mask_edges(edge_index1, node_list, val_prop, test_prop, split_seed)
    data1 = {'edge_index': edge_index1, 'node_list': node_list, 'features': features, 'train_edge':train_edges,
            'train_edge_neg': train_edges_neg, 'val_edge': val_edges, 'val_edge_neg': val_edges_neg, 'test_edge': test_edges,
            'test_edge_neg': test_edges_neg}

    train_edges, train_edges_neg, val_edges, val_edges_neg, test_edges, test_edges_neg = mask_edges(edge_index2, node_list, val_prop, test_prop, split_seed)
    data2 = {'edge_index': edge_index2, 'node_list': node_list, 'features': features, 'train_edge':train_edges,
            'train_edge_neg': train_edges_neg, 'val_edge': val_edges, 'val_edge_neg': val_edges_neg, 'test_edge': test_edges,
            'test_edge_neg': test_edges_neg}

    train_edges, train_edges_neg, val_edges, val_edges_neg, test_edges, test_edges_neg = mask_edges(edge_index3, node_list, val_prop, test_prop, split_seed)
    data3 = {'edge_index': edge_index3, 'node_list': node_list, 'features': features, 'train_edge':train_edges,
            'train_edge_neg': train_edges_neg, 'val_edge': val_edges, 'val_edge_neg': val_edges_neg, 'test_edge': test_edges,
            'test_edge_neg': test_edges_neg}
    
    dataname = ['tree1', 'tree2', 'tree3']
    for data in dataname:
        path = "../data/link_prediction/{}/".format(data)
        if not os.path.exists(path):
            os.makedirs(path)
        filee = path + 'data.pth' 
        if data == 'tree1':
            torch.save(data1, filee)
        elif data == 'tree2':
            torch.save(data2, filee)
        elif data == 'tree3':
            torch.save(data3, filee)

# ############### LINK PREDICTION DATA LOADERS ####################################

def save_dict_to_json(data, filename):
    """Saves a dictionary to a JSON file.

    Args:
        data (dict): The dictionary to save.
        filename (str): The name of the file to save to (must end with .json).
    """
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4) # indent is optional, for pretty formatting


def load_data_lp(dataset, use_feats, data_path, split_seed):
    path = "../data/link_prediction/{}/".format(dataset)
    if not os.path.exists(path):
        os.makedirs(path)
    filee = path + 'data.pth' 
    if os.path.exists(filee):
        data = torch.load(filee)
        return data

    if dataset in ['cora', 'citeseer', 'pubmed']:
        edge_index, node_list, features, _, _, _, _ = load_citation_data(
            dataset, use_feats, data_path, split_seed
        )
    elif dataset in ['airport']:
        edge_index, node_list, features, _, _, _, _ = load_airport_data(dataset, data_path, split_seed)
    elif dataset in ['disease_nc']:
        edge_index, node_list, features, _, _, _, _ = load_disease_data(dataset, data_path, split_seed)
    elif dataset in ['ogbn-products', 'ogbn-papers100M', 'ogbn-arxiv', 'ogbn-proteins']:
        edge_index, node_list, features, _, _, _, _ = load_ogb_data(dataset, data_path)
    elif dataset in ['ogbl-vessel']:
        edge_index, node_list, features, train_edge, valid_edge, test_edge = load_ogl_data(dataset, data_path)
    elif dataset in ['fb100']:
        edge_index, node_list, features, _, _, _, _ = load_fb100_dataset(dataset, data_path)
    elif dataset in ['twitch_gamer']:
        edge_index, node_list, features, _, _, _, _ = load_twitch_gamer_dataset(dataset, data_path)
    elif dataset in ['tree1', 'tree2', 'tree3']:
        create_tree_dataset(split_seed) # it only run for the first time
        data = load_data_lp(dataset, use_feats, data_path, split_seed)
        return data
    else:
        raise Exception('pls provide the dataset')

    val_prop = 0.05
    test_prop = 0.1
    if dataset == 'ogbl-vessel':
        train_edges = train_edge['edge'].t()
        train_edges_neg = train_edge['edge_neg'].t()
        val_edges = valid_edge['edge'].t()
        val_edges_neg = valid_edge['edge_neg'].t()
        test_edges = test_edge['edge'].t()
        test_edges_neg = test_edge['edge_neg'].t()
    else:
        train_edges, train_edges_neg, val_edges, val_edges_neg, test_edges, test_edges_neg = mask_edges(edge_index, node_list, val_prop, test_prop, split_seed)
    data = {'edge_index': edge_index, 'node_list': node_list, 'features': features, 'train_edge':train_edges,
            'train_edge_neg': train_edges_neg, 'val_edge': val_edges, 'val_edge_neg': val_edges_neg, 'test_edge': test_edges,
            'test_edge_neg': test_edges_neg}
    torch.save(data, filee)
    return data

def load_ogl_data(dataset_str, data_path):
    if not os.path.isdir(data_path):
        os.makedirs(data_path)
    
    dataset = PygLinkPropPredDataset(name=dataset_str, root=data_path)
    split_edge = dataset.get_edge_split()
    train_edge, valid_edge, test_edge = split_edge["train"], split_edge["valid"], split_edge["test"]
    graph = dataset[0]
    features = torch.tensor(graph["x"]).float().contiguous() if graph["x"] is not None else None
    edge_index = graph["edge_index"]
    #edge_index = torch.from_numpy(edge_index).to(torch.int)
    num_nodes = graph['num_nodes']
    node_list = list(range(num_nodes))

    return edge_index, node_list, features, train_edge, valid_edge, test_edge

def mask_edges(edge_index, node_list, val_prop, test_prop, seed):
    # edge_index = 2 x E, no self loop
    np.random.seed(seed)  # get tp edges , random negative sampling so the result may be different
    src, dst = edge_index
    # Identify self-loops
    self_loop_mask = src != dst
    # Remove self-loop
    edge_index_no_loops = edge_index[:, self_loop_mask]
    edge_index = edge_index_no_loops
    self_loops = edge_index[0] == edge_index[1]
    assert not self_loops.any()
    m_pos = edge_index.shape[1]
    n_val = int(m_pos * val_prop)
    n_test = int(m_pos * test_prop)
    val_edges, test_edges, train_edges = edge_index[:, :n_val], edge_index[:, n_val:n_test + n_val], edge_index[:, n_test + n_val:]

    def ismember(edge, edge_index, t=True):
        edge_tensor = torch.tensor(edge).view(2, 1)
        if not t:
            edge_index = torch.from_numpy(edge_index).t()
        else:
            edge_index = torch.from_numpy(edge_index)
        matches = (edge_tensor == edge_index).type(torch.LongTensor)
        m = matches[0] + matches[1]
        exists = torch.any(m == 2)
        return exists

    list_neg_edge_index = []
    numpy_edge_index = edge_index.numpy()
    while len(list_neg_edge_index) < m_pos:
        sampled_nodes = np.random.choice(np.array(node_list), 2, replace=True)
        idx_i = sampled_nodes[0]
        idx_j = sampled_nodes[1]
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], numpy_edge_index):
            continue
        if ismember([idx_j, idx_i], numpy_edge_index):
            continue
        if list_neg_edge_index:
            if ismember([idx_j, idx_i], np.array(list_neg_edge_index), False):
                continue
            if ismember([idx_i, idx_j], np.array(list_neg_edge_index), False):
                continue
        list_neg_edge_index.append([idx_i, idx_j])

    print("Done negative sampling!!!!")
    
    neg_edge_index = torch.tensor(list_neg_edge_index).t().type(torch.LongTensor) # 2 x E

    val_edges_neg, test_edges_neg = neg_edge_index[:, :n_val], neg_edge_index[:, n_val:n_test + n_val]
    train_edges_neg = torch.cat([neg_edge_index, val_edges, test_edges], dim=1)

    return train_edges, train_edges_neg, val_edges, val_edges_neg, test_edges, test_edges_neg