import json
import numpy as np
import matplotlib.pyplot as plt

# PLOT BINARY HTENS RESULT

with open('checkpoints/HTENS_nary_ms/checkpoint.json', 'r') as f:
    nary_d = json.load(f)
nary_results = nary_d['results']

with open('checkpoints/HTENS_full_ms/checkpoint.json', 'r') as f:
    full_d = json.load(f)
cancomp_results = full_d['results']

full_nparams = np.array([x['n_cell_param'] for x in cancomp_results]).reshape((3, 5, 5))
nary_nparams = np.array([x['n_cell_param'] for x in nary_results]).reshape((3, 5, 5))

full_best_epoch = np.array([x['best_epoch'] for x in cancomp_results]).reshape((3, 5, 5))
nary_best_epoch = np.array([x['best_epoch'] for x in nary_results]).reshape((3, 5, 5))

full_val_root_acc = np.array([x['root_val'] for x in cancomp_results]).reshape((3, 5, 5))
nary_val_root_acc = np.array([x['root_val'] for x in nary_results]).reshape((3, 5, 5))

full_val_node_acc = np.array([x['nodes_val'] for x in cancomp_results]).reshape((3, 5, 5))
nary_val_node_acc = np.array([x['nodes_val'] for x in nary_results]).reshape((3, 5, 5))

full_tr_time = np.array([x['tr_forw_time'] + x['tr_backw_time'] for x in cancomp_results]).reshape((3, 5, 5))
nary_tr_time = np.array([x['tr_forw_time'] + x['tr_backw_time'] for x in nary_results]).reshape((3, 5, 5))

full_tr_backw_time = np.array([x['tr_backw_time'] for x in cancomp_results]).reshape((3, 5, 5))
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


# PLOT PROD RESULT
with open('checkpoints/PROD_nary_stat_ms/checkpoint.json', 'r') as f:
    nary_d = json.load(f)
nary_results = nary_d['results']

with open('checkpoints/PROD_cancomp_stat_ms/checkpoint.json', 'r') as f:
    cancomp_d = json.load(f)
cancomp_results = cancomp_d['results']

with open('checkpoints/PROD_hosvd_stat_ms/checkpoint.json', 'r') as f:
    hosvd_d = json.load(f)
hosvd_results = hosvd_d['results']

cancomp_nparams = np.array([x['n_cell_param'] for x in cancomp_results]).reshape((3, 5, 5))
nary_nparams = np.array([x['n_cell_param'] for x in nary_results]).reshape((3, 5, 5))
hosvd_nparams = np.array([x['n_cell_param'] for x in hosvd_results]).reshape((3, 5, 5))

cancomp_best_epoch = np.array([x['best_epoch'] for x in cancomp_results]).reshape((3, 5, 5))
nary_best_epoch = np.array([x['best_epoch'] for x in nary_results]).reshape((3, 5, 5))
hosvd_best_epoch = np.array([x['best_epoch'] for x in hosvd_results]).reshape((3, 5, 5))

cancomp_val_root_acc = np.array([x['root_val'] for x in cancomp_results]).reshape((3, 5, 5))
nary_val_root_acc = np.array([x['root_val'] for x in nary_results]).reshape((3, 5, 5))
hosvd_val_root_acc = np.array([x['root_val'] for x in hosvd_results]).reshape((3, 5, 5))

cancomp_val_node_acc = np.array([x['nodes_val'] for x in cancomp_results]).reshape((3, 5, 5))
nary_val_node_acc = np.array([x['nodes_val'] for x in nary_results]).reshape((3, 5, 5))
hosvd_val_node_acc = np.array([x['nodes_val'] for x in hosvd_results]).reshape((3, 5, 5))

cancomp_tr_time = np.array([x['tr_forw_time'] + x['tr_backw_time'] for x in cancomp_results]).reshape((3, 5, 5))
nary_tr_time = np.array([x['tr_forw_time'] + x['tr_backw_time'] for x in nary_results]).reshape((3, 5, 5))
hosvd_tr_time = np.array([x['tr_forw_time'] + x['tr_backw_time'] for x in hosvd_results]).reshape((3, 5, 5))

cancomp_tr_backw_time = np.array([x['tr_backw_time'] for x in cancomp_results]).reshape((3, 5, 5))
nary_tr_backw_time = np.array([x['tr_backw_time'] for x in nary_results]).reshape((3, 5, 5))
hosvd_tr_backw_time = np.array([x['tr_backw_time'] for x in hosvd_results]).reshape((3, 5, 5))

cancomp_idx_max = cancomp_val_root_acc.mean(2).argmax(0)
nary_idx_max = nary_val_root_acc.mean(2).argmax(0)
hosvd_idx_max = hosvd_val_root_acc.mean(2).argmax(0)

plt.figure()
plt.title('Root accuracy on PROD dataset')
plt.xlabel('N params')
plt.ylabel('Root acc.')
plt.semilogx(cancomp_nparams[0,:,0], cancomp_val_root_acc.mean(2).max(0), label='Cancomp', marker='x', fillstyle='none')
plt.semilogx(nary_nparams[0,:,0], nary_val_root_acc.mean(2).max(0), label='Nary', marker='o', fillstyle='none')
plt.semilogx(hosvd_nparams[0,:,0], hosvd_val_root_acc.mean(2).max(0), label='Hosvd', marker='+', fillstyle='none')
plt.legend()

plt.figure()
plt.title('Nodes accuracy on PROD dataset')
plt.xlabel('N params')
plt.ylabel('Node acc.')
plt.semilogx(cancomp_nparams[0,:,0], cancomp_val_node_acc.mean(2).max(0), label='Cancomp', marker='x', fillstyle='none')
plt.semilogx(nary_nparams[0,:,0], nary_val_node_acc.mean(2).max(0), label='Nary', marker='o', fillstyle='none')
plt.semilogx(hosvd_nparams[0,:,0], hosvd_val_node_acc.mean(2).max(0), label='Hosvd', marker='+', fillstyle='none')
plt.legend()

# plt.figure()
# plt.title('Number of epoch to converge to the best result on PROD dataset')
# plt.xlabel('N params')
# plt.ylabel('N epoch')
# plt.semilogx(cancomp_nparams[0, :, 0], cancomp_best_epoch.mean(2)[cancomp_idx_max, range(5)], label='Cancomp', marker='x', fillstyle='none')
# plt.semilogx(nary_nparams[0, :, 0], nary_best_epoch.mean(2)[nary_idx_max, range(5)], label='Nary', marker='o', fillstyle='none')
# plt.legend()

# plt.figure()
# plt.xlabel('N params')
# plt.ylabel('Training time')
# plt.semilogx(cancomp_nparams[0, :, 0], cancomp_tr_time.mean(2)[cancomp_idx_max, range(5)], label='Cancomp', marker='x', fillstyle='none')
# plt.semilogx(nary_nparams[0, :, 0], nary_tr_time.mean(2)[nary_idx_max, range(5)], label='Nary', marker='o', fillstyle='none')
# plt.legend()

# lr_list = [0.01, 0.02, 0.05]
#
# for i in range(3):
#     plt.figure()
#     plt.title('Training time for epoch  on PROD dataset with lr={} '.format(lr_list[i]))
#     plt.xlabel('N params')
#     plt.ylabel('Training time (s)')
#     plt.semilogx(cancomp_nparams[0, :, 0], (cancomp_tr_time / cancomp_best_epoch).mean(2)[i, :], label='Cancomp', marker='x', fillstyle='none')
#     plt.semilogx(nary_nparams[0, :, 0], (nary_tr_time / nary_best_epoch).mean(2)[i, :], label='Nary', marker='o', fillstyle='none')
#     plt.legend()

for i in range(3):
    plt.figure()
    plt.title('Number of epoch to converge on PROD dataset with lr={} '.format(lr_list[i]))
    plt.xlabel('N params')
    plt.ylabel('N epoch')
    plt.semilogx(cancomp_nparams[0, :, 0], cancomp_best_epoch.mean(2)[i, :], label='Cancomp', marker='x', fillstyle='none')
    plt.semilogx(nary_nparams[0, :, 0], nary_best_epoch.mean(2)[i, :], label='Nary', marker='o', fillstyle='none')
    plt.semilogx(hosvd_nparams[0, :, 0], hosvd_best_epoch.mean(2)[i, :], label='Hosvd', marker='+', fillstyle='none')
    plt.legend()
