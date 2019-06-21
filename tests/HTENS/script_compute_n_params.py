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

cancomp_list = []
for i in range(1, 750):
    m = CANCOMPCell(10,2,i,False)
    cancomp_list.append(sum(p.numel() for p in m.parameters() if p.requires_grad))

cancomp_list.insert(0,-1)
cancomp_array = np.array(cancomp_list)

hosvd_list = []
for i in range(1, 10):
    m = HOSVDCell(10,2,i,False)
    hosvd_list.append(sum(p.numel() for p in m.parameters() if p.requires_grad))

hosvd_list.insert(0,-1)
hosvd_array = np.array(hosvd_list)

pair_h = []
pair_val = []
for i in [5,10,20,50,10]:
    ii = find_nearest(nary_array, full_array[i])
    iii = find_nearest(cancomp_array, full_array[i])
    pair_h.append((i,ii,iii))
    pair_val.append((full_array[i], nary_array[ii], cancomp_array[iii]))