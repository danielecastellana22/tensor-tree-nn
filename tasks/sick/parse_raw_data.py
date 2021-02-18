import os
import argparse
from exputils.utils import eprint, path_exists_with_message
from exputils.serialisation import to_pkl_file
from preprocessing.NLP.parsers import NLPAllParser
from tqdm import tqdm


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

    fnames = ['SICK_train.txt', 'SICK_trial.txt', 'SICK_test.txt']

    if not os.path.exists(output_dir):
        eprint('Creating output directory.')
        os.makedirs(output_dir)

    eprint('Start the parsing!')
    parser = NLPAllParser()

    entailment_vocab = {'NEUTRAL': 0, 'ENTAILMENT': 1, 'CONTRADICTION': 2}

    for f_name in fnames:
        rf_name = os.path.join(input_dir, f_name)
        wf_dep_name = os.path.join(output_dir, f_name.replace('.txt', '_dep.pkl'))
        wf_const_name = os.path.join(output_dir, f_name.replace('.txt', '_const.pkl'))
        wf_bin_const_name = os.path.join(output_dir, f_name.replace('.txt', '_bin_const.pkl'))

        if not path_exists_with_message(wf_dep_name) or not path_exists_with_message(wf_const_name) or not path_exists_with_message(wf_bin_const_name):
            ris = {'dep': [], 'const': [], 'bin_const': []}
            with open(rf_name, 'r', encoding='utf-8') as rf:
                skip_first_line = True
                for l in tqdm(rf.readlines(), desc='Buildiing trees from {}: '.format(f_name)):
                    if skip_first_line:
                        skip_first_line = False
                        continue

                    v = l.strip().split('\t')
                    sent_a = v[1]
                    sent_b = v[2]
                    rel_score = float(v[3])
                    ent_judgment = entailment_vocab[v[4]]

                    ris_a, = parser.raw_parse(sent_a)
                    ris_b, = parser.raw_parse(sent_b)
                    for k in ris_a:
                        ris[k].append({'tree_a': ris_a[k],
                                       'tree_b': ris_b[k],
                                       'relatedness': rel_score,
                                       'entailment': ent_judgment})

            eprint('Saving parsed trees.')
            to_pkl_file(ris['dep'], wf_dep_name)
            to_pkl_file(ris['const'], wf_const_name)
            to_pkl_file(ris['bin_const'], wf_bin_const_name)

    words_vocab_file = os.path.join(output_dir, 'words_vocab.pkl')
    if not path_exists_with_message(words_vocab_file):
        eprint('Store word vocabulary.')
        to_pkl_file(parser.words_vocab, words_vocab_file)
