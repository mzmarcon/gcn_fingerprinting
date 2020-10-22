import os
import numpy as np
from collections import defaultdict
import torch
from scipy import sparse
from scipy.sparse.linalg.eigen.arpack import eigsh
import torch.nn.functional as F
import torch.nn as nn
from numba import cuda

def get_adjacency(cn_matrix, threshold):
    # mask = (cn_matrix > np.percentile(cn_matrix, threshold)).astype(np.uint8)
    mask = (cn_matrix > threshold).astype(np.uint8)
    sparse_matrix = cn_matrix * mask
    nodes, neighbors = np.nonzero(mask)
    sparse_indices = {}
    for i, node in enumerate(nodes):
        #remove self-loops of indices dict
        if not neighbors[i] == node:
            if not node in sparse_indices: 
                sparse_indices[node] = [neighbors[i]]
            else:
                sparse_indices[node].append(neighbors[i])
    return sparse_matrix, sparse_indices

def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    # print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sparse.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sparse.eye(adj.shape[0])

    t_k = list()
    t_k.append(sparse.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sparse.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for _ in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)
    # return sparse.coo_matrix(t_k)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sparse.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sparse.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sparse.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def select_pairs(batch_data):
    """
    Find pairs within a mini-batch for nt_xent loss.
    """
    ids = [item[0] for item in batch_data['input_anchor']['id']] + [item[0] for item in batch_data['input_positive']['id']] + [item[0] for item in batch_data['input_negative']['id']]
    pairs=[]
    for i,_ in enumerate(ids): 
        pairs.append([j for j,_ in enumerate(ids) if ids[i]==ids[j]])   
    return pairs


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        label = label[0]
        loss_list = []
        for n in range(len(output1)):
            euclidean_distance = F.pairwise_distance(output1[n], output2[n])
            loss_contrastive = torch.mean((1-label[n]) * torch.pow(euclidean_distance, 2) +
                                        (label[n]) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
            loss_list.append(loss_contrastive)
        
        loss = torch.mean(torch.stack(loss_list,dim=0))

        return loss


class ContrastiveCosineLoss(torch.nn.Module):
    """
    Based on NT_Xent Loss from SimCLR. 
    """

    def __init__(self, temperature=1.0):
        super(ContrastiveCosineLoss, self).__init__()
        self.temperature = temperature

    def nt_xent(self, output, anchor_n, pair_pos, pair_neg):
        dist = nn.CosineSimilarity()    
        loss = []
        for pos_item in pair_pos:
            #compute sum of similarities between each positive pair and all negatives pairs
            neg_sim = torch.sum(torch.stack([torch.exp(dist(output[pos_item].unsqueeze(0),output[neg_item].unsqueeze(0))/self.temperature) for neg_item in pair_neg]))
            pos_sim = torch.exp(dist(output[anchor_n].unsqueeze(0),output[pos_item].unsqueeze(0))/self.temperature)    
            nt_xent = -1 * torch.log(pos_sim / (pos_sim + neg_sim))
            loss.append(nt_xent)
        
        return torch.mean(torch.stack(loss))

    def forward(self, output_anchor, output_pos, output_neg, pairs):
        loss_batch = []
        output = torch.cat((output_anchor, output_pos, output_neg))
        set_size = set(range(output.shape[0]))
        for n in range(len(output)):
            pair_pos = set(pairs[0]).difference([n])
            pair_neg = set_size.difference(pairs[n])

            loss = self.nt_xent(output,n,pair_pos,pair_neg)
            loss_batch.append(loss)
        
        loss = torch.mean(torch.stack(loss_batch,dim=0))

        return loss


