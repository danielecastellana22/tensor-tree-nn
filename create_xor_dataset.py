import os
import numpy as np
from tqdm import tqdm
from treeLSTM import *

# TODO: move to another file
class MyTreeDataset(BracketTreeDataset):

    def __init__(self, path_dir, file_name):
        BracketTreeDataset.__init__(self, path_dir, file_name)
        self.__load_trees__()

    def __build_tree__(self, root):
        a = 4

    def get_loader(self, batch_size, device, shuffle=False):
        return 3


class MyNode:

    def __init__(self):
        self.label = 0
        self.left = None
        self.right = None


def to_bracket_representation(root):
        if root is None:
            return ""
        return "({} {} {})".format(root.label, to_bracket_representation(root.left), to_bracket_representation(root.right))


def create_random_tree(root, cont, n_node_max):

    if cont < n_node_max:
        if np.random.rand() > 0.5:
            # the node has a left child
            lt = MyNode()
            root.left = lt
            cont += 1
            create_random_tree(lt, cont, n_node_max)

    if cont < n_node_max:
        if np.random.rand() > 0.5:
            # the node has a left child
            rt = MyNode()
            root.right = rt
            cont += 1
            create_random_tree(rt, cont, n_node_max)

    if root.left is None and root.right is None:
        root.label = 1 if np.random.rand() > 0.5 else 0

    if root.left is not None and root.right is None:
        root.label = root.left.label

    if root.left is None and root.right is not None:
        root.label = root.right.label

    if root.left is not None and root.right is not None:
        root.label = 1 if root.left.label == root.right.label else 0

def main(N, n_max, save_dir):
    trees = []
    train_filename = os.path.join(save_dir, 'train.txt')
    dev_filename = os.path.join(save_dir, 'dev.txt')
    test_filename = os.path.join(save_dir, 'test.txt')
    for i in tqdm(range(N), desc='Creating trees: '):
        r = MyNode()
        create_random_tree(r, 1, n_max)
        trees.append(r)

    with open(train_filename,'w') as trainfile:
        for i in tqdm(range(0, int(0.7*N)), desc='Writing train file: '):
            trainfile.write(to_bracket_representation(trees[i]))

    with open(dev_filename,'w') as devfile:
        for i in tqdm(range(int(0.7*N), int(0.8*N)), desc='Writing dev file: '):
            devfile.write(to_bracket_representation(trees[i]))

    with open(test_filename,'w') as testile:
        for i in tqdm(range(int(0.8*N), N), desc='Writing test file: '):
            testile.write(to_bracket_representation(trees[i]))


#TODO: add parsing argument
if __name__ == '__main__':
    #main(10000, 100, 'data/xor/')

    br = MyTreeDataset('data/xor/','train.txt')