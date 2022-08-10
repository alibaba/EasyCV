# Copyright (c) Alibaba, Inc. and its affiliates.
import torch


def LpDistance(query_emb, ref_emb, p=2):
    """
    Input:
        query_emb: [n, dims] tensor
        ref_emb: [m, dims] tensor
        p : p normalize
    Output:
        distance_matrix: [n, m] tensor

    distance_matrix_i_j = (\sigma_k(a_i_k**p - b_j_k**p))**(1/p)
    """
    return torch.cdist(query_emb, ref_emb, p)


def DotproductSimilarity(query_emb, ref_emb):
    return torch.einsum('ik,jk->ij', [query_emb, ref_emb])


def CosineSimilarity(query_emb, ref_emb):
    """
    Input:
        query_emb: [n, dims] tensor
        ref_emb: [m, dims] tensor
    Output:
        distance_matrix: [n, m] tensor
    """
    a = torch.nn.functional.normalize(query_emb, p=2, dim=1)
    b = torch.nn.functional.normalize(ref_emb, p=2, dim=1)

    return torch.einsum('ik,jk->ij', [a, b])
