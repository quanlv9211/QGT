import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F

from script.utils.eval_utils import acc_f1

class NCModel(nn.Module):
    """
    Base model for node classification task.
    """

    def __init__(self, args):
        super(NCModel, self).__init__()

        if args.n_classes > 2:
            self.f1_average = 'micro'
        else:
            self.f1_average = 'binary'
        self.pos_weight = args.pos_weight
        self.n_classes = args.n_classes
        self.device = args.device
        

    def decode(self, embeddings, idx):
        return F.log_softmax(embeddings[idx], dim=1)

    def compute_metric_loss(self, output, labels, idx):
        if self.pos_weight:
            weights = torch.Tensor([1., 1. / labels[idx].mean()] * self.n_classes ).to(self.device)
        else:
            weights = torch.Tensor([1.] * self.n_classes).to(self.device) 
        loss = F.nll_loss(output, labels[idx], weights)
        acc, f1 = acc_f1(output, labels[idx], average=self.f1_average)
        metrics = {'loss': loss, 'acc': acc, 'f1': f1}
        return metrics

    def init_metric_dict(self):
        return {'acc': -1, 'f1': -1}

    def has_improved(self, m1, m2):
        return m1["f1"] < m2["f1"]


class LPModel(nn.Module):
    def __init__(self, args):
        super(LPModel, self).__init__()
        self.r = 2.0
        self.t = 1.0
        self.sigmoid = True
        self.model = args.model
        self.use_hyperdecoder = args.use_hyperdecoder and args.model in ['QGCN2', 'QGT']

    @staticmethod
    def maybe_num_nodes(index, num_nodes=None):
        return index.max().item() + 1 if num_nodes is None else num_nodes

    def decoder(self, dist):
        # value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        # return torch.sigmoid(value) if sigmoid else value
        return torch.sigmoid(dist) if self.sigmoid else dist

    def hyperdeoder(self, dist): ## cho nay co van de
        def FermiDirac(dist):
            probs = 1. / (torch.exp((dist - self.r) / self.t) + 1.0)
            return probs

        # edge_i = edge_index[0]
        # edge_j = edge_index[1]
        # z_i = torch.nn.functional.embedding(edge_i, z)
        # z_j = torch.nn.functional.embedding(edge_j, z)
        # dist = manifold.sqdist(z_i, z_j, factor)
        return FermiDirac(dist)

    def compute_loss(self, dist_pos, dist_neg=None):
        decoder = self.hyperdeoder if self.use_hyperdecoder else self.decoder
        pos_loss = -torch.log(decoder(dist_pos) + 1e-15).mean()
        if dist_neg is not None:
            neg_loss = -torch.log(1 - decoder(dist_neg) + 1e-15).mean()
        else:
            neg_loss = 0

        return pos_loss + neg_loss

    def compute_metric(self, dist_pos, dist_neg, pos_edge_index, neg_edge_index):
        decoder = self.hyperdeoder if self.use_hyperdecoder else self.decoder
        pos_y = torch.ones(pos_edge_index.shape[1])
        neg_y = torch.zeros(neg_edge_index.shape[1])
        y = torch.cat([pos_y, neg_y], dim=0)
        pos_pred = decoder(dist_pos)
        neg_pred = decoder(dist_neg)
        pred = torch.cat([pos_pred, neg_pred], dim=0)
        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
        auc = roc_auc_score(y, pred)
        ap = average_precision_score(y, pred)
        metric = {'auc': auc, 'ap': ap}
        return metric

    def init_metric_dict(self):
        return {'auc': -1, 'ap': -1}

    def has_improved(self, m1, m2):
        return m1["auc"] < m2["auc"]