import os
from nltk import Tree
from tqdm import tqdm


def bin_to_nary(t:Tree):
    if isinstance(t,str):
        # is a label
        return t

    if len(t)==2:
        t0 = bin_to_nary(t[0])
        t1 = bin_to_nary(t[1])
    else:
        t0 = Tree(t.label()[1:], [])
        t1 = bin_to_nary(t[0])

    if isinstance(t1, str):
        if t1 == ']':
            return t0

    t0.append(t1)

    return t0


def load_trees(f_name):

    t_data = []
    with open(os.path.join(data_dir, f_name), 'r') as txtfile:
        for sent in tqdm(txtfile.readlines(), desc='Loading trees: '):
            l_sent = sent[:-1].split('\t')

            if l_sent[1][0] == '(':
                a_t = Tree.fromstring(l_sent[1])
                a_t = bin_to_nary(a_t)
                t_data.append((l_sent[0], a_t))
            else:
                t_data.append((l_sent[0], l_sent[1]))

    return t_data


def write_trees(fname, data):
    m = 100000000000000000
    with open(os.path.join(data_dir,fname),'w') as f:
        for v1,v2 in data:
            if isinstance(v2, Tree):
                v2 = v2.pformat(margin=m)
            f.write(v1+'\t'+v2+'\n')


if __name__ == '__main__':
    data_dir = 'data/ListOpsPaper'

    tr_data = load_trees('train_d20s.tsv')
    test_data = load_trees('test_d20s.tsv')

    dev_data = tr_data[80000:]
    tr_data = tr_data[:80000]

    write_trees('train.txt', tr_data)
    write_trees('dev.txt', dev_data)
    write_trees('test.txt', test_data)