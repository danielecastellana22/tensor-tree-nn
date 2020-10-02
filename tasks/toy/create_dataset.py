import argparse
import os
import numpy as np
from tqdm import tqdm
from utils.utils import get_logger
import functools


class MyNode:

    def __init__(self):
        self.label = None
        self.children = []
        self.word = None


def to_bracket_representation(root, print_labels):
        if root is None:
            return ""

        if not root.children:
            # leaf node
            s = " {}".format(root.word)
        else:
            # internal node
            s = '('
            if print_labels:
                s += '{}_{} '.format(root.label, root.word)
            else:
                s += '{} '.format(root.word)

            for t in root.children:
                ss = to_bracket_representation(t, print_labels)
                s += ss + ' '
            s += ')'
        return s


def get_input_data_string(tree, label_only_on_root):
    s = ''
    if label_only_on_root:
        s += '{}\t'.format(tree.label)
        s += to_bracket_representation(tree, False)
    else:
        s += to_bracket_representation(tree, True)
    s += '\n'
    return s


def create_fun_assign_markov(mat_list, in_vocab, max_output_degree):
    h_size = np.size(list(in_vocab.values())[0])

    def f(root):
        if not root.children:
            i = np.random.randint(0,len(in_vocab))
            root.word = list(in_vocab.keys())[i]
            root.h = in_vocab[root.word]
            root.label = 1 if np.sum(root.h) > 0 else 0
        else:

            ris = None
            for i in range(max_output_degree):
                if i < len(root.children):
                    h = root.children[i].h
                else:
                    h = np.zeros(h_size)
                h = np.append(h, [1])

                m = mat_list[i]

                if ris is None:
                    ris = np.matmul(m, h)
                else:
                    # apply both ris and h
                    aux = m.reshape((-1, m.shape[2]))
                    aux = np.matmul(aux, h).reshape((-1, m.shape[1]))
                    ris = np.append(ris, [1])
                    ris = np.matmul(aux, ris)

            root.h = np.tanh(ris.squeeze())
            root.label = 1 if np.sum(root.h) > 0 else 0

    return f


def create_fun_assign_full(A, in_vocab, max_output_degree):
    h_size = np.size(list(in_vocab.values())[0])

    def f(root):
        if not root.children:
            i = np.random.randint(0,len(in_vocab))
            root.word = list(in_vocab.keys())[i]
            root.h = in_vocab[root.word]
            root.label = 1 if np.sum(root.h) > 0 else 0
        else:
            ris = A.reshape((A.shape[0], -1))
            for i in range(max_output_degree):
                if i < len(root.children):
                    h = root.children[i].h
                else:
                    h = np.zeros(h_size)

                h = np.append(h, [1])
                ris = np.matmul(h, ris).reshape((A.shape[i+1], -1))

            root.h = np.tanh(ris.squeeze())
            root.label = 1 if np.sum(root.h) > 0 else 0

    return f


def create_fun_assign_op_on_list(in_vocab_dict, ops_dict):

    n_ops = len(ops_dict)
    ops_name = list(ops_dict.keys())

    n_vocabs = len(in_vocab_dict)
    n_vocabs_name = list(in_vocab_dict.keys())

    def f(root):
        if not root.children:
            voc_id = np.random.randint(n_vocabs)
            voc_k = n_vocabs_name[voc_id]

            root.label = in_vocab_dict[voc_k]
            root.word = voc_k
            root.h = root.label
        else:
            in_list = []
            for t in root.children:
                in_list.append(t.h)

            op_id = np.random.randint(n_ops)
            op_k = ops_name[op_id]
            op = ops_dict[op_k]

            root.h = op(in_list)
            root.label = root.h
            root.word = op_k

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


def write_vocabularies_file(f_name, list_vocs):

    with open(f_name, 'w') as f:
        for d in list_vocs:
            s = ''
            for v in d:
                s+= str(v)+'\t'
            s = s[:-1] + '\n'
            f.write(s)


def main(args):
    np.seterr(all='raise')

    trees = []
    if args.dataset_name != '':
        dir = os.path.join(args.output_dir, 'toy_{}_{}'.format(args.type, args.dataset_name))
    else:
        dir = os.path.join(args.output_dir, 'toy_{}'.format(args.type))

    if not os.path.exists(dir):
        os.makedirs(dir)

    log =  get_logger('runner', dir, file_name='creation_dataset.log', write_on_console=True)

    aux_filename = os.path.join(dir, 'tensors.txt')
    train_filename = os.path.join(dir, 'train.txt')
    dev_filename = os.path.join(dir, 'dev.txt')
    test_filename = os.path.join(dir, 'test.txt')
    voc_filename = os.path.join(dir, 'vocabs.txt')

    log.info(str(args))

    if args.type == 'full':
        sz_A = [args.h_size+1 for x in range(args.max_output_degree)]
        sz_A.append(args.h_size)
        A = np.random.randn(*sz_A)
        in_vocab = {'a': np.random.randn(args.h_size), 'b': np.random.randn(args.h_size), 'c': np.random.randn(args.h_size)}

        with open(aux_filename, 'w') as ff:
            for k in in_vocab:
                ff.write('{}:\n {}\n'.format(str(k), str(in_vocab[k])))
            ff.write('A:\n {}\n'.format(str(A)))

        assign_label_fun = create_fun_assign_full(A, in_vocab, args.max_output_degree)

    elif args.type == 'markov':
        mat_list = []
        for i in range(args.max_output_degree):
            if i==0:
                m = np.random.randn(args.h_size+1, args.h_size)
            else:
                m = np.random.randn(args.h_size+1, args.h_size+1, args.h_size)
            mat_list.append(m)
        in_vocab = {'a': np.random.randn(args.h_size), 'b': np.random.randn(args.h_size), 'c': np.random.randn(args.h_size)}

        with open(aux_filename, 'w') as ff:
            for k in in_vocab:
                ff.write('{}:\n {}\n'.format(str(k), str(in_vocab[k])))
            for i in range(args.max_output_degree):
                ff.write('mat {}:\n {}\n'.format(i, str(mat_list[i])))

        assign_label_fun = create_fun_assign_markov(mat_list, in_vocab, args.max_output_degree)

    elif args.type == 'prod':
        ops_dict = {'PROD': np.prod}
        in_vocab_dict = {'1': 1, '-1': -1}
        write_vocabularies_file(voc_filename, [in_vocab_dict.keys(), in_vocab_dict.keys(), ops_dict.keys()])
        assign_label_fun = create_fun_assign_op_on_list(in_vocab_dict, ops_dict)

    elif args.type == 'minmax':
        ops_dict = {'MIN': np.min, 'MAX': np.max}
        in_vocab_dict = {str(x): x for x in range(10)}
        write_vocabularies_file(voc_filename, [in_vocab_dict.keys(), in_vocab_dict.keys(), ops_dict.keys()])
        assign_label_fun = create_fun_assign_op_on_list(in_vocab_dict, ops_dict)

    elif args.type == 'med':
        ops_dict = {'MED': lambda l: int(np.floor(np.median(l)))}
        in_vocab_dict = {str(x): x for x in range(10)}
        write_vocabularies_file(voc_filename, [in_vocab_dict.keys(), in_vocab_dict.keys(), ops_dict.keys()])
        assign_label_fun = create_fun_assign_op_on_list(in_vocab_dict, ops_dict)

    elif args.type == 'select':
        assign_label_fun = create_fun_assign_op_on_list(0, 10, [lambda l: l[np.sum(l) % len(l)]])

    elif args.type == 'vote':
        assign_label_fun = create_fun_assign_op_on_list(0, 10, [lambda l: np.argmax(np.bincount(l))])

    elif args.type == 'sorted':
        assign_label_fun = create_fun_assign_op_on_list(0, 1, [lambda l: int(np.all(np.diff(l)>=0))+1])

    elif args.type == 'bool':
        ops_dict = {'AND': lambda l: int(np.all(l)),
                    'OR': lambda l: int(np.any(l)),
                    'XOR': lambda l: int(np.logical_xor.reduce(l)),
                    'IMPL': lambda l: int(functools.reduce(lambda x, y: np.logical_or(np.logical_not(x), y), l, True))}
        in_vocab_dict = {'0': 0, '1': 1}
        write_vocabularies_file(voc_filename, [in_vocab_dict.keys(), in_vocab_dict.keys(), ops_dict.keys()])
        assign_label_fun = create_fun_assign_op_on_list(in_vocab_dict, ops_dict)

    else:
        raise ValueError('Type not known.')

    print_only_root_label = not args.all_labels

    count = {}
    tot_nodes = 0
    for i in tqdm(range(args.N), desc='Creating trees: '):
        r = MyNode()
        tot_nodes += create_random_tree(r, 1, args.min_height, args.max_height, args.max_output_degree, assign_label_fun)

        if r.label not in count:
            count[r.label] = 0
        count[r.label] += 1

        trees.append(r)

    log.info('{} nodes have been generated.'.format(tot_nodes))

    for k in sorted(count):
        log.info('{} trees of type {}'.format(count[k], k))

    with open(train_filename, 'w') as trainfile:
        for i in tqdm(range(0, int(0.7*args.N)), desc='Writing train file: '):
            trainfile.write(get_input_data_string(trees[i], print_only_root_label))

    with open(dev_filename, 'w') as devfile:
        for i in tqdm(range(int(0.7*args.N), int(0.8*args.N)), desc='Writing dev file: '):
            devfile.write(get_input_data_string(trees[i], print_only_root_label))

    with open(test_filename, 'w') as testile:
        for i in tqdm(range(int(0.8*args.N), args.N), desc='Writing test file: '):
            testile.write(get_input_data_string(trees[i], print_only_root_label))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=10000)
    parser.add_argument('--min-height', type=int, default=4)
    parser.add_argument('--max-height', type=int, default=8)
    parser.add_argument('--type', default='htens')
    parser.add_argument('--output-dir', default='data')
    parser.add_argument('--h-size', type=int, default=2)
    parser.add_argument('--dataset-name', default="")
    parser.add_argument('--max-output-degree', type=int, default=5)
    parser.add_argument('--all-labels', default=False, action='store_true')
    args = parser.parse_args()
    # print(args)
    main(args)