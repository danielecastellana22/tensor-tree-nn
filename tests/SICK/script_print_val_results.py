import json
import numpy as np
#import matplotlib.pyplot as plt



with open('checkpoints/SICK_cancomp_stat_ms/checkpoint.json', 'r') as f:
    cancomp_d = json.load(f)

cancomp_results = cancomp_d['results']


with open('checkpoints/SICK_hosvd_stat_ms/checkpoint.json', 'r') as f:
    full_hosvd = json.load(f)

hosvd_results = full_hosvd['results']


MSE_val_cc = np.array([float(x['MSE_val'][7:13]) for x in d['results']]).reshape((3,3,3,5))
Pearson_val_cc = np.array([float(x['Pearson_val'][7:13]) for x in d['results']]).reshape((3,3,3,5))

lr_list = [0.01, 0.02, 0.05]
rank_list_cancomp = [70, 100, 150]
weight_decay_list = [1e-5, 1e-4, 1e-3]

best_MSE_wrt_rank_cc = MSE_val_cc.mean(3).min(2).min(0)
best_Pearson_wrt_rank_cc = Pearson_val_cc.mean(3).max(2).max(0)


with open('checkpoints/SICK_hosvd_stat_ms/checkpoint.json', 'r') as f:
    d = json.load(f)

MSE_val_hs = np.array([float(x['MSE_val'][7:13]) for x in d['results']]).reshape((3,2,3,5))
Pearson_val_hs = np.array([float(x['Pearson_val'][7:13]) for x in d['results']]).reshape((3,2,3,5))

lr_list = [0.01, 0.02, 0.05]
rank_list_hosvd = [2, 3]
weight_decay_list = [1e-5, 1e-4, 1e-3]

best_MSE_wrt_rank_hs = MSE_val_hs.mean(3).min(2).min(0)
best_Pearson_wrt_rank_hs = Pearson_val_hs.mean(3).max(2).max(0)

#plt.plot(list(map(count_full_params, h_vals_f)), vl_best_h_f)
#plt.plot(list(map(count_full_params, h_vals_f)), tr_best_h_f)

#plt.figure()

#plt.plot(wd, vl_best_wd_f)
#plt.plot(wd, tr_best_wd_f)

with open('checkpoints/SICK_nary_stat_ms/checkpoint.json', 'r') as f:
    nary_d = json.load(f)
nary_results = nary_d['results']

nary_mse = np.array([x['MSE_val'] for x in nary_results]).reshape((2, 5))
nary_pearson = np.array([x['Pearson_val'] for x in nary_results]).reshape((2, 5))
