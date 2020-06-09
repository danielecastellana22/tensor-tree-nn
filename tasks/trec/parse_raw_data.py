import os
import argparse
from utils.utils import eprint, path_exists_with_message
from utils.serialization import to_pkl_file, to_json_file, from_pkl_file
from preprocessing.NLP.parsers import NLPAllParser
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split

def parse_arguments():
    parser = argparse.ArgumentParser()
    # experiment params
    parser.add_argument('input_dir', help='Directory which contains raw data.')
    parser.add_argument('output_dir', help='Directory where to store the results.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    input_dir = args.input_dir
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        eprint('Creating output directory.')
        os.makedirs(output_dir)

    eprint('Start the parsing!')
    parser = NLPAllParser()

    coarse_label_vocab = {}
    fine_label_vocab = {}

    file_to_parse = {'train': 'train_5500.label.txt',
                     'test': 'TREC_10.label.txt'}

    all_parsed_trees = {}
    for k_t, v in file_to_parse.items():
        rf = os.path.join(input_dir, v)
        out_file = os.path.join(output_dir, 'all_parsed_trees_{}.pkl'.format(k_t))
        if not path_exists_with_message(out_file):
            parsed_trees = {'dep': [], 'const': [], 'bin_const': []}
            with open(rf, 'r', encoding='utf-8') as rf:
                for l in tqdm(rf.readlines(), desc='Buildiing trees from {}: '.format(rf)):

                    v = l.strip().split(' ', maxsplit=1)
                    fine_label = v[0]
                    sent = v[1]
                    coarse_label = fine_label.split(':')[0]

                    coarse_label_id = coarse_label_vocab.setdefault(coarse_label, len(coarse_label_vocab))
                    fine_label_id = fine_label_vocab.setdefault(fine_label, len(fine_label_vocab))

                    ris, = parser.raw_parse(sent)
                    for kk in ris:
                        parsed_trees[kk].append({'tree': ris[kk],
                                                'coarse_label': coarse_label_id,
                                                'fine_label': fine_label_id})
            to_pkl_file(parsed_trees, out_file)

            # save words vocab file
            eprint('Store word vocabulary.')
            words_vocab_file = os.path.join(output_dir, 'words_vocab.pkl')
            to_pkl_file(parser.words_vocab, words_vocab_file)

            # strore label vocabs
            eprint('Store label vocabulary.')
            to_json_file(coarse_label_vocab, os.path.join(output_dir, 'coarse_vocab.json'))
            to_json_file(fine_label_vocab, os.path.join(output_dir, 'fine_vocab.json'))
        else:
            parsed_trees = from_pkl_file(out_file)

        all_parsed_trees[k_t] = parsed_trees

    # compute validation split
    # rand_perm_idx = np.random.permutation(len(parsed_trees['dep']))
    # idx_val = rand_perm_idx[:n_val]
    # idx_train = rand_perm_idx[n_val:]
    n_trees = len(all_parsed_trees['train']['dep'])
    n_val = 500
    labels = [d['coarse_label'] for d in all_parsed_trees['train']['dep']]
    idx_train, idx_val = train_test_split(np.arange(n_trees), test_size=n_val / n_trees, stratify=labels)

    trees_to_write = {'train': {k: (np.array(v)[idx_train]).tolist() for k, v in all_parsed_trees['train'].items()},
                      'validation': {k: (np.array(v)[idx_val]).tolist() for k, v in all_parsed_trees['train'].items()},
                      'test': all_parsed_trees['test']}

    eprint('Saving parsed trees.')
    for k, v in trees_to_write.items():
        for kk, vv in v.items():
            wf_name = os.path.join(output_dir, '{}_{}.pkl'.format(k, kk))
            to_pkl_file(vv, wf_name)