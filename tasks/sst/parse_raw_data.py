import os
import argparse
from utils.misc import eprint, path_exists_with_message
from utils.serialization import to_pkl_file
from preprocessing.NLP.parsers import NLPAllParser
from collections import OrderedDict
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

    #fname_sent_map = 'sentiment_map.txt'
    fname_phrase_id = 'dictionary.txt'
    fname_id_sentiment = 'sentiment_labels.txt'
    fnames = ['train.txt', 'validation.txt', 'test.txt']

    if not os.path.exists(output_dir):
        eprint('Creating output directory.')
        os.makedirs(output_dir)

    eprint('Start the parsing!')
    parser = NLPAllParser()
    tokenizer_prop = {'tokenize.options': 'splitAssimilations=false'}

    # build sentiment map
    sentiment_map_out_file = os.path.join(output_dir, 'sentiment_map.pkl')
    if not path_exists_with_message(sentiment_map_out_file):
        id2sentiment = []
        skip_first_line = True
        with open(os.path.join(input_dir, fname_id_sentiment), 'r', encoding='utf-8') as f_sentiment_id:
            for l in tqdm(f_sentiment_id.readlines(), desc='Loading sentiment labels: '):
                if skip_first_line:
                    skip_first_line = False
                    continue

                v = l.split('|')
                float_label = float(v[1])
                if float_label <= 0.2:
                    int_label = 0
                elif float_label <= 0.4:
                    int_label = 1
                elif float_label <= 0.6:
                    int_label = 2
                elif float_label <= 0.8:
                    int_label = 3
                else:
                    int_label = 4

                assert int(v[0]) == len(id2sentiment)
                id2sentiment.append(int_label)

        sentiment_map = OrderedDict()
        with open(os.path.join(input_dir, fname_phrase_id), 'r', encoding='utf-8') as f_phrase_id:
            for l in tqdm(f_phrase_id.readlines(), desc='Creating sentiment map: '):
                v = l.split('|')
                txt = v[0]
                id = int(v[1])
                if ' ' in txt:
                    tok_list = [x.lower() for x in parser.tokenize(txt, tokenizer_prop)]
                else:
                    tok_list = [txt.lower()]

                tok_set = tuple(tok_list)

                if tok_set not in sentiment_map:
                    sentiment_map[tok_set] = id2sentiment[id]
                else:
                    if sentiment_map[tok_set] == 2:  # neutral
                        sentiment_map[tok_set] = id2sentiment[id]

        # store sentiment map
        eprint('Saving sentiment map.')
        to_pkl_file(sentiment_map, sentiment_map_out_file)

    for f_name in fnames:
        rf_name = os.path.join(input_dir, f_name)
        wf_dep_name = os.path.join(output_dir, f_name.replace('.txt', '_dep.pkl'))
        wf_const_name = os.path.join(output_dir, f_name.replace('.txt', '_const.pkl'))
        wf_bin_const_name = os.path.join(output_dir, f_name.replace('.txt', '_bin_const.pkl'))

        if not path_exists_with_message(wf_dep_name) or not path_exists_with_message(wf_const_name) or not path_exists_with_message(wf_bin_const_name):
            ris = {'dep': [], 'const': [], 'bin_const': []}
            with open(rf_name, 'r', encoding='utf-8') as rf:
                for l in tqdm(rf.readlines(), desc='Buildiing trees from {}: '.format(f_name)):
                    ris_p, = parser.raw_parse(l, tokenizer_prop)
                    for k in ris_p:
                        ris[k].append(ris_p[k])

            eprint('Saving parsed trees.')
            to_pkl_file(ris['dep'], wf_dep_name)
            to_pkl_file(ris['const'], wf_const_name)
            to_pkl_file(ris['bin_const'], wf_bin_const_name)

    words_vocab_file = os.path.join(output_dir, 'words_vocab.pkl')
    if not path_exists_with_message(words_vocab_file):
        eprint('Store word vocabulary.')
        to_pkl_file(parser.words_vocab, words_vocab_file)
