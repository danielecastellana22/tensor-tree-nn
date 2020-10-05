import os
import argparse
from utils.misc import eprint, path_exists_with_message
from utils.serialization import to_pkl_file
from preprocessing.tree_conversions import string_to_nltk_tree
from tqdm import tqdm
import json
from preprocessing.NLP.parsers import NLPAllParser


def parse_arguments():
    parser = argparse.ArgumentParser()
    # experiment params
    parser.add_argument('input_dir', help='Directory which contains raw data.')
    parser.add_argument('output_dir', help='Directory where to store the results.')
    return parser.parse_args()


def update_words_vocab(w_vocab, nx_t):
    w_list = [d['word'] for u, d in nx_t.nodes(data=True) if 'word' in d]
    for w in w_list:
        k = w.lower()
        id = len(w_vocab)
        w_vocab.setdefault(k, id)


if __name__ == '__main__':
    args = parse_arguments()

    input_dir = args.input_dir
    output_dir = args.output_dir

    fnames = ['snli_1.0_train.jsonl', 'snli_1.0_dev.jsonl', 'snli_1.0_test.jsonl']

    if not os.path.exists(output_dir):
        eprint('Creating output directory.')
        os.makedirs(output_dir)
    else:
        eprint('Output dir already exists!')
        exit(1)

    entailment_vocab = {'neutral': 0, 'entailment': 1, 'contradiction': 2}
    words_vocab = {}

    for f_name in fnames:
        rf_name = os.path.join(input_dir, f_name)
        with open(rf_name, 'r', encoding='utf-8') as rf:

            ris = {'const': [], 'bin_const': []}
            c = 0
            for i,l in enumerate(tqdm(rf.readlines(), desc='Reading trees from {}: '.format(f_name))):
                json_d = json.loads(l)

                nltk_a = string_to_nltk_tree(json_d['sentence1_parse'])
                const_a = NLPAllParser.__const_tree_to_nx__(nltk_a)
                nltk_a.chomsky_normal_form()
                bin_const_a = NLPAllParser.__const_tree_to_nx__(nltk_a)
                nltk_b = string_to_nltk_tree(json_d['sentence2_parse'])
                const_b = NLPAllParser.__const_tree_to_nx__(nltk_b)
                nltk_b.chomsky_normal_form()
                bin_const_b = NLPAllParser.__const_tree_to_nx__(nltk_b)

                # update words vocab
                update_words_vocab(words_vocab, const_a)
                update_words_vocab(words_vocab, const_b)

                if json_d['gold_label'] in entailment_vocab:
                    ent_judgment = entailment_vocab[json_d['gold_label']]
                    ris['const'].append({'tree_a': const_a,
                                         'tree_b': const_b,
                                         'entailment': ent_judgment})

                    ris['bin_const'].append({'tree_a': bin_const_a,
                                             'tree_b': bin_const_b,
                                             'entailment': ent_judgment})

                if (i+1) % (5*(10**4)) == 0:
                    eprint('Saving parsed trees.')
                    wf_const_name = os.path.join(output_dir, f_name.replace('.jsonl', '_{}_const.pkl'.format(c)))
                    wf_bin_const_name = os.path.join(output_dir, f_name.replace('.jsonl', '_{}_bin_const.pkl'.format(c)))
                    to_pkl_file(ris['const'], wf_const_name)
                    to_pkl_file(ris['bin_const'], wf_bin_const_name)
                    c = c + 1
                    ris = {'const': [], 'bin_const': []}

            eprint('Saving parsed trees.')
            wf_const_name = os.path.join(output_dir, f_name.replace('.jsonl', '_{}_const.pkl'.format(c)))
            wf_bin_const_name = os.path.join(output_dir, f_name.replace('.jsonl', '_{}_bin_const.pkl'.format(c)))
            to_pkl_file(ris['const'], wf_const_name)
            to_pkl_file(ris['bin_const'], wf_bin_const_name)

    words_vocab_file = os.path.join(output_dir, 'words_vocab.pkl')
    if not path_exists_with_message(words_vocab_file):
        eprint('Store word vocabulary.')
        to_pkl_file(words_vocab, words_vocab_file)
