import argparse
import os
import numpy as np
from tqdm import tqdm


class MyNode:

    def __init__(self):
        self.label = 0
        self.left = None
        self.right = None
        self.word = None


def to_bracket_representation(root):
        if root is None:
            return ""
        if root.left is None and root.right is None:
            return "({} {})".format(root.label, root.word)

        return "({} {} {})".format(root.label, to_bracket_representation(root.left), to_bracket_representation(root.right))


def create_fun_assign_HTENS(A, h_size):
    def f(root):

        if root.left is None and root.right is None:
            root.label = 1 if np.random.rand() > 0.5 else 0
            root.word = 'a' if root.label==1 else 'b'
            root.h = np.ones((h_size)) if root.label == 1 else -np.ones((h_size))

        if root.left is not None and root.right is None:
            root.label = root.left.label
            root.h = root.left.h

        if root.left is None and root.right is not None:
            root.label = root.right.label
            root.h = root.right.h

        if root.left is not None and root.right is not None:
            root.h = np.einsum('ijk,j,k->i',A,root.left.h,root.right.h)
            root.label = 1 if np.sum(root.h) > 0 else 0
    return f


def assign_XOR_label(root):
    if root.left is None and root.right is None:
        root.label = 1 if np.random.rand() > 0.5 else 0
        root.word = 'a' if root.label==1 else 'b'

    if root.left is not None and root.right is None:
        root.label = root.left.label

    if root.left is None and root.right is not None:
        root.label = root.right.label

    if root.left is not None and root.right is not None:
        root.label = 1 if root.left.label == root.right.label else 0


def assign_NOT_LEFT_label(root):
    if root.left is None and root.right is None:
        root.label = 1 if np.random.rand() > 0.5 else 0
        root.word = 'a' if root.label==1 else 'b'

    if root.left is not None and root.right is None:
        root.label = root.left.label

    if root.left is None and root.right is not None:
        root.label = root.right.label

    if root.left is not None and root.right is not None:
        root.label = 1 if root.left.label == 1 else 0


def create_random_tree(root, h, h_min, h_max, assign_label_fun):

    if h < h_min:
        # create both child
        root.left = MyNode()
        root.right = MyNode()
        create_random_tree(root.left, h + 1, h_min, h_max, assign_label_fun)
        create_random_tree(root.right, h + 1, h_min, h_max, assign_label_fun)
    else:
        v = np.random.randint(h_max-h)
        if v > 0:
            # create both child
            root.left = MyNode()
            root.right = MyNode()
            create_random_tree(root.left, h + 1, h_min, h_max, assign_label_fun)
            create_random_tree(root.right, h + 1, h_min, h_max, assign_label_fun)

    assign_label_fun(root)


def main(args):
    trees = []
    train_filename = os.path.join(args.save_dir, 'train.txt')
    dev_filename = os.path.join(args.save_dir, 'dev.txt')
    test_filename = os.path.join(args.save_dir, 'test.txt')

    if args.type == 'xor':
        assigning_function =  assign_XOR_label
    elif args.type == 'notl':
        assigning_function = assign_NOT_LEFT_label
    elif args.type == 'htens':
        A = np.random.randn(args.h_size, args.h_size, args.h_size)
        with open(os.path.join(args.save_dir,'log.txt'),'w') as ff:
            ff.write(str(A))
        assigning_function = create_fun_assign_HTENS(A, args.h_size)
    else:
        raise ValueError('Type not known.')

    for i in tqdm(range(args.N), desc='Creating trees: '):
        r = MyNode()
        create_random_tree(r, 1, args.min_height, args.max_height, assigning_function)
        trees.append(r)

    with open(train_filename, 'w') as trainfile:
        for i in tqdm(range(0, int(0.7*args.N)), desc='Writing train file: '):
            trainfile.write(to_bracket_representation(trees[i])+'\n')

    with open(dev_filename, 'w') as devfile:
        for i in tqdm(range(int(0.7*args.N), int(0.8*args.N)), desc='Writing dev file: '):
            devfile.write(to_bracket_representation(trees[i])+'\n')

    with open(test_filename, 'w') as testile:
        for i in tqdm(range(int(0.8*args.N), args.N), desc='Writing test file: '):
            testile.write(to_bracket_representation(trees[i])+'\n')


#TODO: add parsing argument
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=10000)
    parser.add_argument('--min-height', type=int, default=4)
    parser.add_argument('--max-height', type=int, default=6)
    parser.add_argument('--type', default='htens')
    parser.add_argument('--save-dir', default='data/htens')
    parser.add_argument('--h-size', type=int, default=10)
    args = parser.parse_args()
    # print(args)
    main(args)