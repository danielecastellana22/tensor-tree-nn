import pickle
import numpy as np
import matplotlib.pyplot as plt


def count_full_params(h):
    return h**3 + 2*h**2


def count_nary_params(h):
    return 2*h**2


def h_nary_same_params(h_full):
    return np.sqrt(count_full_params(h_full)/2)


h_vals_n = [10, 25, 66, 255, 714, 937, 1308, 2010]
h_vals_f = [5, 10, 20, 50, 100, 120, 150, 200]
size_ris = [3, 8, 5]

with open('checkpoints/nary_HTENS_ModelSelection_mul/checkpoint.pickle', 'rb') as f:
    d = pickle.load(f)
ris = d['results']
vl_metric = np.array([x['vl_acc'] for x in ris]).reshape([3, 5, 5])
tr_metric = np.array([x['tr_acc'] for x in ris]).reshape([3, 5, 5])
with open('checkpoints/nary_HTENS_ModelSelection_mul2/checkpoint.pickle', 'rb') as f:
    d = pickle.load(f)
ris2 = d['results']
vl_metric2 = np.array([x['vl_acc'] for x in ris2]).reshape([3, 3, 5])
tr_metric2 = np.array([x['tr_acc'] for x in ris2]).reshape([3, 3, 5])
vl_metric = np.concatenate([vl_metric, vl_metric2], axis=1)
tr_metric = np.concatenate([tr_metric, tr_metric2], axis=1)
vl_avg_best_h_n = np.average(vl_metric, axis=2)
vl_std_best_h_n = np.std(vl_metric, axis=2)
vl_best_h_n = np.max(vl_avg_best_h_n, axis=0)
vl_best_err_h_n = vl_std_best_h_n[np.argmax(vl_avg_best_h_n, axis=0),np.arange(8)]
tr_avg_best_h_n = np.average(tr_metric, axis=2)
tr_std_best_h_n = np.std(tr_metric, axis=2)
tr_best_h_n = np.max(tr_avg_best_h_n, axis=0)
tr_best_err_h_n = tr_std_best_h_n[np.argmax(tr_avg_best_h_n, axis=0),np.arange(8)]

with open('checkpoints/full_HTENS_ModelSelection_mul/checkpoint.pickle', 'rb') as f:
    d = pickle.load(f)
ris = d['results']
vl_metric = np.array([x['vl_acc'] for x in ris]).reshape([3, 5, 5])
tr_metric = np.array([x['tr_acc'] for x in ris]).reshape([3, 5, 5])
with open('checkpoints/full_HTENS_ModelSelection_mul2/checkpoint.pickle', 'rb') as f:
    d = pickle.load(f)
ris2 = d['results']
vl_metric2 = np.array([x['vl_acc'] for x in ris2]).reshape([3, 3, 5])
tr_metric2 = np.array([x['tr_acc'] for x in ris2]).reshape([3, 3, 5])
vl_metric = np.concatenate([vl_metric, vl_metric2], axis=1)
tr_metric = np.concatenate([tr_metric, tr_metric2], axis=1)
vl_avg_best_h_f = np.average(vl_metric, axis=2)
vl_std_best_h_f = np.std(vl_metric, axis=2)
vl_best_h_f = np.max(vl_avg_best_h_f, axis=0)
vl_best_err_h_f = vl_std_best_h_f[np.argmax(vl_avg_best_h_f, axis=0),np.arange(8)]
tr_avg_best_h_f = np.average(tr_metric, axis=2)
tr_std_best_h_f = np.std(tr_metric, axis=2)
tr_best_h_f = np.max(tr_avg_best_h_f, axis=0)
tr_best_err_h_f = tr_std_best_h_f[np.argmax(tr_avg_best_h_f, axis=0),np.arange(8)]

plt.plot(list(map(count_full_params,h_vals_f)), vl_best_h_f)
plt.plot(list(map(count_nary_params,h_vals_n)), vl_best_h_n)
plt.plot(list(map(count_full_params,h_vals_f)), tr_best_h_f)
plt.plot(list(map(count_nary_params,h_vals_n)), tr_best_h_n)