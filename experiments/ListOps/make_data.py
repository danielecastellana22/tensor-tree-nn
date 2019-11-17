import random
import numpy as np
import os
import argparse
from tqdm import tqdm

START = '['
MIN = "MIN"
MAX = "MAX"
MED = "MED"
FIRST = "FIRST"
LAST = "LAST"
SUM_MOD = "SM"
END = "]"

OPERATORS = [MIN, MAX, MED, SUM_MOD]  # , FIRST, LAST]
VALUES = range(10)

VALUE_P = 0.25
MAX_ARGS = 5
MAX_DEPTH = 20

DATA_POINTS = 100000


def generate_tree(depth, binary=True):
    if depth < MAX_DEPTH:
        r = random.random()
    else:
        r = 1

    if depth > 1 and r > VALUE_P:
        value = random.choice(VALUES)
        return value
    else:
        op = random.choice(OPERATORS)
        num_values = random.randint(2, MAX_ARGS)
        values = []
        for _ in range(num_values):
            values.append(generate_tree(depth + 1, binary))

        if binary:
            t = (op+START, values[0])
            for value in values[1:]:
                t = (t, value)
            t = (t, END)
        else:
            t = (op, *values)#, END)

        return t


def to_string(t, parens=True, binary=True):
    if isinstance(t, str):
        return t
    elif isinstance(t, int):
        return str(t)
    else:
        if parens:
            s = '('
        else:
            s= ''

        for ch in t:
            s += to_string(ch, parens, binary) + ' '

        if parens:
            s += ')'

        return s


def to_value(t, binary=True):
    if binary:
        if not isinstance(t, tuple):
            return t
        l = to_value(t[0], binary)
        r = to_value(t[1], binary)
        if l in OPERATORS:  # Create an unsaturated function.
            return (l, [r])
        elif r == END:  # l must be an unsaturated function.
            if l[0] == MIN:
                return min(l[1])
            elif l[0] == MAX:
                return max(l[1])
            elif l[0] == FIRST:
                return l[1][0]
            elif l[0] == LAST:
                return l[1][-1]
            elif l[0] == MED:
                return int(np.median(l[1]))
            elif l[0] == SUM_MOD:
                return np.sum(l[1]) % 10
        elif isinstance(l, tuple):  # We've hit an unsaturated function and an argument.
            return l[0], l[1] + [r]
    else:
        if not isinstance(t, tuple):
            return t
        else:
            op = t[0]
            arg_list = []
            for ch in t[1:len(t)]:
                arg_list.append(to_value(ch, binary))

            if op == MIN:
                return min(arg_list)
            elif op == MAX:
                return max(arg_list)
            elif op == FIRST:
                return arg_list[0]
            elif op == LAST:
                return arg_list[-1]
            elif op == MED:
                return int(np.median(arg_list))
            elif op == SUM_MOD:
                return np.sum(arg_list) % 10


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # experiment params
    parser.add_argument('--data-dir', default='data/ListOps')
    parser.add_argument('--max-depth', default=MAX_DEPTH, type=int)
    parser.add_argument('--max-args', default=MAX_ARGS, type=int)
    parser.add_argument('--n-examples', default=DATA_POINTS, type=int)
    parser.add_argument('--binary', default=False, type=bool)
    parser.add_argument('--tr-split', default=0.7, type=float)
    parser.add_argument('--dev-split', default=0.1, type=float)
    parser.add_argument('--test-split', default=0.2, type=float)

    args = parser.parse_args()

    MAX_DEPTH = args.max_depth
    MAX_ARGS = args.max_args
    DATA_POINTS = args.n_examples

    #create dir
    if not os.path.exists(args.data_dir):
        os.mkdir(args.data_dir)

    data = set()
    prev_len = 0
    with tqdm(desc='Generating trees: ', total=DATA_POINTS) as pbar:
        while len(data) < DATA_POINTS:
            data.add(generate_tree(1, binary=args.binary))
            pbar.update(len(data) - prev_len)
            prev_len = len(data)

    n_tr = int(DATA_POINTS * args.tr_split)
    n_dev = int(DATA_POINTS * args.dev_split)
    n_test = int(DATA_POINTS * args.test_split)

    data = list(data)
    tr_data = data[:n_tr]
    dev_data = data[n_tr:n_tr+n_dev]
    test_data = data[n_tr+n_dev:]

    with open(os.path.join(args.data_dir, 'train.txt'), 'w', encoding='utf-8') as f_tr:
        for example in tqdm(tr_data, desc='Writing training trees: '):
            f_tr.write(str(to_value(example, binary=args.binary)) + '\t' +
                       to_string(example, parens=True, binary=args.binary) + '\n')

    with open(os.path.join(args.data_dir, 'dev.txt'), 'w', encoding='utf-8') as f_dev:
        for example in tqdm(dev_data, desc='Writing validation trees: '):
            f_dev.write(str(to_value(example, binary=args.binary)) + '\t' +
                        to_string(example, parens=True, binary=args.binary) + '\n')

    with open(os.path.join(args.data_dir, 'test.txt'), 'w', encoding='utf-8') as f_test:
        for example in tqdm(test_data, desc='Writing test trees: '):
            f_test.write(str(to_value(example, binary=args.binary)) + '\t' +
                         to_string(example, parens=True, binary=args.binary) + '\n')