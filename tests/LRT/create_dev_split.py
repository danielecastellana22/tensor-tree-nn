import os
if __name__ == '__main__':
    mypath = 'data/lrt'
    tr_files = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f)) and f.startswith('train')]

    for f_name in tr_files:
        with open(f_name,'r') as f:
            for n, l in enumerate(f):
                pass
            n=n+1

            dev_n = int(n*0.1)

            # TODO: choose randomly dev_n lines
            # TODO: moves this lines in a dev file
