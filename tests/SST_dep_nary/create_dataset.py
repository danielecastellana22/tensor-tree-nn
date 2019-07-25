from nltk.parse.corenlp import CoreNLPDependencyParser
import os
from tqdm import tqdm

if __name__ == '__main__':
    mypath = 'data/sst_nary/string'
    writepath = 'data/sst_nary/dep_tree'
    file_names = ['train.txt', 'validation.txt', 'test.txt']
    fname_sent_map = 'sentiment_map.txt'
    fname_phrase_id = 'dictionary.txt'
    fname_id_sentiment = 'sentiment_labels.txt'

    dep_parser = CoreNLPDependencyParser(url='http://localhost:9000')

    for f_name in file_names:
        rf_name = os.path.join(mypath, f_name)
        wf_name = os.path.join(writepath, f_name)
        if not os.path.exists(wf_name):
            with open(rf_name, 'r',encoding='utf-8') as rf,\
                 open(wf_name, 'w', encoding='utf-8') as wf:
                for l in tqdm(rf.readlines(), desc='Buildiing dependency trees from {}: '.format(f_name)):
                    a, = dep_parser.raw_parse(l)
                    wf.write(a.tree()._pformat_flat(nodesep='', parens='()', quotes=False) + '\n')
        else:
            print('{} already exists.'.format(wf_name))

    # build the sentiment map
    if not os.path.exists(os.path.join(writepath, fname_sent_map)):
        id2sentiment = []
        skip = True
        with open(os.path.join(mypath, fname_id_sentiment), 'r', encoding='utf-8') as f_sentiment_id:
            for l in tqdm(f_sentiment_id.readlines(), desc='Loading sentiment labels: '):

                if skip:
                    skip = False
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

                id2sentiment.append(int_label)

        phrase2sentiment = {}
        with open(os.path.join(mypath, fname_phrase_id), 'r', encoding='utf-8') as f_phrase_id,\
             open(os.path.join(writepath, fname_sent_map), 'w', encoding='utf-8') as f_sent_map:
            for l in tqdm(f_phrase_id.readlines(), desc='Loading dictionary: '):
                v = l.split('|')
                txt = v[0]
                id = int(v[1])
                sent = id2sentiment[id]
                tok_key = list(dep_parser.tokenize(txt, properties={'tokenize.options': 'normalizeParentheses=True'}))
                f_sent_map.write('{}|{}\n'.format(' '.join(tok_key), sent))
    else:
        print('{} already exists.'.format(os.path.join(writepath, fname_sent_map)))