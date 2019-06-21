import argparse
import os
import numpy as np
from tqdm import tqdm


class MyNode:

    def __init__(self):
        self.label = 0
        self.children = []
        self.word = None


def to_bracket_representation(root):
        if root is None:
            return ""
        if not root.children:
            return "({} {})".format(root.label, root.word)

        s = '({} '.format(root.label)
        for t in root.children:
            ss = to_bracket_representation(t)
            s += ss + ' '
        s += ')'
        return s


def create_fun_assign_PROD(h_size):
    def f(root):
        if not root.children:
            root.label = 1 if np.random.rand() > 0.5 else 0
            root.word = 'a' if root.label==1 else 'b'
            root.h = np.ones((h_size)) if root.label == 1 else -np.ones((h_size))
        else:
            ris = np.ones(h_size)
            for t in root.children:
                ris *= t.h

            root.h = ris.squeeze()
            root.label = 1 if np.sum(root.h) > 0 else 0

    return f


def create_fun_assign_HTENS(A, h_size, max_output_degree):
    def f(root):
        if not root.children:
            root.label = 1 if np.random.rand() > 0.5 else 0
            root.word = 'a' if root.label==1 else 'b'
            root.h = np.ones((h_size)) if root.label == 1 else -np.ones((h_size))
        else:
            ris = A.reshape((A.shape[0], -1))
            for i in range(max_output_degree):
                if i < len(root.children):
                    h = root.children[i].h
                else:
                    h = np.zeros(h_size)

                h = np.append(h, [1])
                ris = np.matmul(h, ris).reshape((A.shape[i+1], -1))

            root.h = ris.squeeze()
            root.label = 1 if np.sum(root.h) > 0 else 0

    return f


def create_random_tree(root, h, h_min, h_max, max_output_degree, assign_label_fun):

    if h < h_min:
        n_child = np.random.randint(1, max_output_degree + 1)
    else:
        v = np.random.randint(h_max-h)
        if v > 0:
            n_child = np.random.randint(0, max_output_degree + 1)
        else:
            n_child = 0

    n_nodes = 0
    for i in range(n_child):
        t = MyNode()
        root.children.append(t)
        n_nodes += create_random_tree(t, h + 1, h_min, h_max, max_output_degree, assign_label_fun)

    assign_label_fun(root)

    return n_nodes + 1

def main(args):
    trees = []

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    train_filename = os.path.join(args.save_dir, 'train.txt')
    dev_filename = os.path.join(args.save_dir, 'dev.txt')
    test_filename = os.path.join(args.save_dir, 'test.txt')

    if args.type == 'htens':
        sz_A = [args.h_size+1 for x in range(args.max_output_degree)]
        sz_A.append(args.h_size)
        A = np.random.randn(*sz_A)
        with open(os.path.join(args.save_dir,'log.txt'),'w') as ff:
            ff.write(str(A))
        assigning_function = create_fun_assign_HTENS(A, args.h_size, args.max_output_degree)
    elif args.type == 'prod':
        assigning_function = create_fun_assign_PROD(args.h_size)
    else:
        raise ValueError('Type not known.')

    count = [0, 0]
    tot_nodes = 0
    for i in tqdm(range(args.N), desc='Creating trees: '):
        r = MyNode()
        tot_nodes += create_random_tree(r, 1, args.min_height, args.max_height, args.max_output_degree, assigning_function)
        count[r.label] += 1
        trees.append(r)

    print('\n{} nodes have been generated.'.format(tot_nodes))
    print('{} trees of type 0 and {} trees of type 1 have been generated.'.format(*count))

    with open(train_filename, 'w') as trainfile:
        for i in tqdm(range(0, int(0.7*args.N)), desc='Writing train file: '):
            trainfile.write(to_bracket_representation(trees[i])+'\n')

    with open(dev_filename, 'w') as devfile:
        for i in tqdm(range(int(0.7*args.N), int(0.8*args.N)), desc='Writing dev file: '):
            devfile.write(to_bracket_representation(trees[i])+'\n')

    with open(test_filename, 'w') as testile:
        for i in tqdm(range(int(0.8*args.N), args.N), desc='Writing test file: '):
            testile.write(to_bracket_representation(trees[i])+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=3000)
    parser.add_argument('--min-height', type=int, default=4)
    parser.add_argument('--max-height', type=int, default=6)
    parser.add_argument('--type', default='prod')
    parser.add_argument('--save-dir', default='data/prod')
    parser.add_argument('--h-size', type=int, default=4)
    parser.add_argument('--max-output-degree', type=int, default=6)
    args = parser.parse_args()
    # print(args)
    main(args)