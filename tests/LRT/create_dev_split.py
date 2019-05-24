import os
import numpy as np


if __name__ == '__main__':
    mypath = 'data/lrt'
    tr_files = [os.path.join(mypath,f) for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f)) and f.startswith('train')]

    for f_name in tr_files:
        with open(f_name, 'r') as f:
            lines = f.readlines()
            n = len(lines)
            tr_n = int(n*0.9)
            np.random.shuffle(lines)

        # write the first 90% of the lines as training
        with open(f_name, 'w') as f:
            for i in range(tr_n):
                f.write(lines[i])
        # write the last 10% of the lines as validation
        dev_name = f_name.replace('train', 'dev')
        with open(dev_name, 'w') as f:
            for i in range(tr_n+1, n):
                f.write(lines[i])
