import pickle
import numpy as np
import matplotlib.pyplot as plt


def count_full_params(h):
    return h**3 + 2*h**2


def count_nary_params(h):
    return 2*h**2


def h_nary_same_params(h_full):
    return np.sqrt(count_full_params(h_full)/2)


h_vals = [5,10,20,50,100,150,200]

with open('checkpoints/nary_HTENS_ModelSelection_sing/checkpoint.pickle','rb') as f:
    d = pickle.load(f)
ris = d['results']
metric = np.array([x['vl_acc'] for x in ris]).reshape([3,7])
best_h_n = np.max(metric,axis=0)

with open('checkpoints/full_HTENS_ModelSelection_sing/checkpoint.pickle','rb') as f:
    d = pickle.load(f)
ris = d['results']
metric = np.array([x['vl_acc'] for x in ris]).reshape([3,7])
best_h_f = np.max(metric, axis=0)