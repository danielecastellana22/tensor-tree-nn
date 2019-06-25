import numpy as np
from treeLSTM.cells import *

def find_nearest(array, value):
    return (np.abs(array - value)).argmin()


full_list = []
for i in range(1,101):
    m = BinaryFullTensorCell(i,2,False)
    full_list.append(sum(p.numel() for p in m.parameters() if p.requires_grad))
full_list.insert(0,-1)
full_array = np.array(full_list)

nary_list = []
for i in range(1,750):
    m = NaryCell(i,2,False)
    nary_list.append(sum(p.numel() for p in m.parameters() if p.requires_grad))

nary_list.insert(0,-1)
nary_array = np.array(nary_list)

hosvd_list = []
for i in range(1, 10):
    m = HOSVDCell(10,2,i,False)
    hosvd_list.append(sum(p.numel() for p in m.parameters() if p.requires_grad))

hosvd_list.insert(0,-1)
hosvd_array = np.array(hosvd_list)

# pair_h = []
# pair_val = []
# for i in [5,10,20,50,10]:
#     ii = find_nearest(nary_array, full_array[i])
#     iii = find_nearest(cancomp_array, full_array[i])
#     pair_h.append((i,ii,iii))
#     pair_val.append((full_array[i], nary_array[ii], cancomp_array[iii]))

cancomp_list = []
sum_list = []
hosvd_list = []
for h in range(1, 100):
    r = h // 2
    m = CANCOMPCell(h, 6, r, True)
    cancomp_list.append(sum(p.numel() for p in m.parameters() if p.requires_grad))
    m = NaryCell(h, 6, True)
    sum_list.append(sum(p.numel() for p in m.parameters() if p.requires_grad))

    rr = []
    for r in range(1,6):
        m = HOSVDCell(h, 6, r, pos_stationarity=True)
        rr.append(sum(p.numel() for p in m.parameters() if p.requires_grad))
    hosvd_list.append(rr)

cancomp_list.insert(0, -1)
cancomp_array = np.array(cancomp_list)
sum_list.insert(0, -1)
sum_array = np.array(sum_list)
hosvd_list.insert(0,[-1, -1, -1,-1 ,-1])
hosvd_array = np.array(hosvd_list)