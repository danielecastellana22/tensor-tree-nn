import json
import numpy as np
import matplotlib.pyplot as plt

with open('checkpoints/HTENS_nary_ms/checkpoint.json', 'r') as f:
    nary_d = json.load(f)

nary_results = nary_d['results'][:25]


with open('checkpoints/HTENS_full_ms/checkpoint.json', 'r') as f:
    full_d = json.load(f)

full_results = full_d['results'][:25]

full_nparams = np.array([x['n_cell_param'] for x in full_results]).reshape((1, 5, 5))
nary_nparams = np.array([x['n_cell_param'] for x in nary_results]).reshape((1, 5, 5))

full_best_epoch = np.array([x['best_epoch'] for x in full_results]).reshape((1, 5, 5))
nary_best_epoch = np.array([x['best_epoch'] for x in nary_results]).reshape((1, 5, 5))

full_val_root_acc = np.array([x['root_val'] for x in full_results]).reshape((1, 5, 5))
nary_val_root_acc = np.array([x['root_val'] for x in nary_results]).reshape((1, 5, 5))

full_val_node_acc = np.array([x['nodes_val'] for x in full_results]).reshape((1, 5, 5))
nary_val_node_acc = np.array([x['nodes_val'] for x in nary_results]).reshape((1, 5, 5))

plt.figure()
plt.xlabel('N params')
plt.ylabel('Root Acc.')
plt.semilogx(full_nparams[0,:,0], full_val_root_acc.mean(2).max(0), label='Full')
plt.semilogx(nary_nparams[0,:,0], nary_val_root_acc.mean(2).max(0), label='Nary')
plt.legend()

plt.figure()
plt.xlabel('N params')
plt.ylabel('Node Acc.')
plt.semilogx(full_nparams[0,:,0], full_val_node_acc.mean(2).max(0), label='Full')
plt.semilogx(nary_nparams[0,:,0], nary_val_node_acc.mean(2).max(0), label='Nary')
plt.legend()

plt.figure()
plt.xlabel('N params')
plt.ylabel('Best epoch')
plt.semilogx(full_nparams[0, :, 0], full_best_epoch.mean(2).max(0), label='Full')
plt.semilogx(nary_nparams[0, :, 0], nary_best_epoch.mean(2).max(0), label='Nary')
plt.legend()