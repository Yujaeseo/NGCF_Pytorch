import numpy as np


def precision_at_k(r, k):
    assert k >= 1
    r = np.asarray(r)[:k]
    return np.mean(r)


def recall_at_k(r, k, all_pos_num):
    r = np.asfarray(r)[:k]
    return np.sum(r) / all_pos_num

def dcg_at_k(r, k, method=1):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.

def ndcg_at_k(r, k, ground_truth, method=1):
    gt = set(ground_truth)
    if len(gt) > k :
        sent_list = [1.0] * k
    else:
        sent_list = [1.0]*len(gt) + [0.0]*(k-len(gt))
    dcg_max = dcg_at_k(sent_list, k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max