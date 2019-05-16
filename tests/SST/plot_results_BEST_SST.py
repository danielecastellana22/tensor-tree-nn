import pickle
import numpy as np
import matplotlib.pyplot as plt


def count_full_params(h):
    return h**3 + 2*h**2


def count_nary_params(h):
    return 2*h**2


def h_nary_same_params(h_full):
    return np.sqrt(count_full_params(h_full)/2)


h_vals_f = [50, 100, 120]
lr_rate = [0.01, 0.02, 0.05]
wd = [1e-4, 1e-3, 1e-2]
size_ris = [3, 8, 5]

vl_metric = None
tr_metric = None
for i in range(1,4):
    with open('checkpoints/full_SST_ModelSelection_best{}/checkpoint.pickle'.format(i), 'rb') as f:
        d = pickle.load(f)
    aux = d['results']
    if vl_metric is None:
        vl_metric = np.array([x['vl_acc'] for x in aux]).reshape([3, 3, 1, 5])
    else:
        vl_metric = np.concatenate((vl_metric, np.array([x['vl_acc'] for x in aux]).reshape([3, 3, 1, 5])), axis=2)

    if tr_metric is None:
        tr_metric = np.array([x['tr_acc'] for x in aux]).reshape([3, 3, 1, 5])
    else:
        tr_metric = np.concatenate((tr_metric, np.array([x['tr_acc'] for x in aux]).reshape([3, 3, 1, 5])), axis=2)

vl_avg_best_h_f = np.average(vl_metric, axis=3)
vl_std_best_h_f = np.std(vl_metric, axis=3)
vl_best_h_f = np.max(vl_avg_best_h_f, axis=(0,2))
vl_best_wd_f = np.max(vl_avg_best_h_f, axis=(0,1))
#vl_best_err_h_f = vl_std_best_h_f[np.argmax(vl_avg_best_h_f, axis=0),np.arange(8)]
tr_avg_best_h_f = np.average(tr_metric, axis=3)
tr_std_best_h_f = np.std(tr_metric, axis=3)
tr_best_h_f = np.max(tr_avg_best_h_f, axis=(0,2))
tr_best_wd_f = np.max(tr_avg_best_h_f, axis=(0,1))
#tr_best_err_h_f = tr_std_best_h_f[np.argmax(tr_avg_best_h_f, axis=0),np.arange(8)]

plt.plot(list(map(count_full_params,h_vals_f)), vl_best_h_f)
plt.plot(list(map(count_full_params,h_vals_f)), tr_best_h_f)

plt.figure()

plt.plot(wd, vl_best_wd_f)
plt.plot(wd, tr_best_wd_f)