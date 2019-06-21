import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

with open('checkpoints/HTENS_nary_ms/checkpoint.json', 'r') as f:
    nary_d = json.load(f)

nary_results = nary_d['results']


with open('checkpoints/HTENS_full_ms/checkpoint.json', 'r') as f:
    full_d = json.load(f)

full_results = full_d['results']

full_nparams = np.array([x['n_cell_param'] for x in full_results]).reshape((3, 5, 5))
nary_nparams = np.array([x['n_cell_param'] for x in nary_results]).reshape((3, 5, 5))

full_best_epoch = np.array([x['best_epoch'] for x in full_results]).reshape((3, 5, 5))
nary_best_epoch = np.array([x['best_epoch'] for x in nary_results]).reshape((3, 5, 5))

full_val_root_acc = np.array([x['root_val'] for x in full_results]).reshape((3, 5, 5))
nary_val_root_acc = np.array([x['root_val'] for x in nary_results]).reshape((3, 5, 5))

full_val_node_acc = np.array([x['nodes_val'] for x in full_results]).reshape((3, 5, 5))
nary_val_node_acc = np.array([x['nodes_val'] for x in nary_results]).reshape((3, 5, 5))

full_tr_time = np.array([x['tr_forw_time'] + x['tr_backw_time'] for x in full_results]).reshape((3, 5, 5))
nary_tr_time = np.array([x['tr_forw_time'] + x['tr_backw_time'] for x in nary_results]).reshape((3, 5, 5))

full_tr_backw_time = np.array([x['tr_backw_time'] for x in full_results]).reshape((3, 5, 5))
nary_tr_backw_time = np.array([x['tr_backw_time'] for x in nary_results]).reshape((3, 5, 5))

full_idx_max = full_val_root_acc.mean(2).argmax(0)
nary_idx_max = nary_val_root_acc.mean(2).argmax(0)

plt.figure()
plt.title('Root accuracy on HTENS dataset')
plt.xlabel('N params')
plt.ylabel('Root acc.')
plt.semilogx(full_nparams[0,:,0], full_val_root_acc.mean(2).max(0), label='Full', marker='x', fillstyle='none')
plt.semilogx(nary_nparams[0,:,0], nary_val_root_acc.mean(2).max(0), label='Nary', marker='o', fillstyle='none')
plt.legend()

plt.figure()
plt.title('Nodes accuracy on HTENS dataset')
plt.xlabel('N params')
plt.ylabel('Node acc.')
plt.semilogx(full_nparams[0,:,0], full_val_node_acc.mean(2).max(0), label='Full', marker='x', fillstyle='none')
plt.semilogx(nary_nparams[0,:,0], nary_val_node_acc.mean(2).max(0), label='Nary', marker='o', fillstyle='none')
plt.legend()

# plt.figure()
# plt.title('Number of epoch to converge to the best result on HTENS dataset')
# plt.xlabel('N params')
# plt.ylabel('N epoch')
# plt.semilogx(full_nparams[0, :, 0], full_best_epoch.mean(2)[full_idx_max, range(5)], label='Full', marker='x', fillstyle='none')
# plt.semilogx(nary_nparams[0, :, 0], nary_best_epoch.mean(2)[nary_idx_max, range(5)], label='Nary', marker='o', fillstyle='none')
# plt.legend()

# plt.figure()
# plt.xlabel('N params')
# plt.ylabel('Training time')
# plt.semilogx(full_nparams[0, :, 0], full_tr_time.mean(2)[full_idx_max, range(5)], label='Full', marker='x', fillstyle='none')
# plt.semilogx(nary_nparams[0, :, 0], nary_tr_time.mean(2)[nary_idx_max, range(5)], label='Nary', marker='o', fillstyle='none')
# plt.legend()

lr_list = [0.01, 0.02, 0.05]

for i in range(3):
    plt.figure()
    plt.title('Training time for epoch  on HTENS dataset with lr={} '.format(lr_list[i]))
    plt.xlabel('N params')
    plt.ylabel('Training time (s)')
    plt.semilogx(full_nparams[0, :, 0], (full_tr_time / full_best_epoch).mean(2)[i, :], label='Full', marker='x', fillstyle='none')
    plt.semilogx(nary_nparams[0, :, 0], (nary_tr_time / nary_best_epoch).mean(2)[i, :], label='Nary', marker='o', fillstyle='none')
    plt.legend()

for i in range(3):
    plt.figure()
    plt.title('Number of epoch to converge on HTENS dataset with lr={} '.format(lr_list[i]))
    plt.xlabel('N params')
    plt.ylabel('N epoch')
    plt.semilogx(full_nparams[0, :, 0], full_best_epoch.mean(2)[i, :], label='Full', marker='x', fillstyle='none')
    plt.semilogx(nary_nparams[0, :, 0], nary_best_epoch.mean(2)[i, :], label='Nary', marker='o', fillstyle='none')
    plt.legend()
