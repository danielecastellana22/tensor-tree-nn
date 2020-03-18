import os
import networkx as nx
import pickle as pkl
from tqdm import tqdm
from preprocessing.utils import ConstValues, load_embeddings
from preprocessing.tree_conversions import string_to_nltk_tree, nx_to_dgl
from utils.utils import eprint
import copy


def build_sentiment_map(input_folder, output_folder, parser):
    fname_sent_map = 'sentiment_map.txt'
    fname_phrase_id = 'dictionary.txt'
    fname_id_sentiment = 'sentiment_labels.txt'
    if os.path.exists(os.path.join(output_folder, fname_sent_map)):
        print('{} already exists.'.format(os.path.join(output_folder, fname_sent_map)))
    else:
        id2sentiment = []
        skip = True
        with open(os.path.join(input_folder, fname_id_sentiment), 'r', encoding='utf-8') as f_sentiment_id:
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

        with open(os.path.join(input_folder, fname_phrase_id), 'r', encoding='utf-8') as f_phrase_id, \
                open(os.path.join(output_folder, fname_sent_map), 'w', encoding='utf-8') as f_sent_map:
            for l in tqdm(f_phrase_id.readlines(), desc='Loading dictionary: '):
                v = l.split('|')
                txt = v[0]
                id = int(v[1])
                sent = id2sentiment[id]

                if ' ' in txt:
                    tok_key = list(parser.tokenize(txt, properties={
                        'tokenize.options': 'normalizeParentheses=True, splitAssimilations=False'}))
                    f_sent_map.write('{}|{}\n'.format(' '.join(tok_key), sent))
                else:
                    f_sent_map.write('{}|{}\n'.format(txt.replace('(', '-LRB-').replace(')', '-RRB-'), sent))


def build_trees(input_folder, output_folder, file_names, parser):
    print('Start buiding trees.')

    for f_name in file_names:
        rf_name = os.path.join(input_folder, f_name)
        wf_name = os.path.join(output_folder, f_name.replace('.txt', '.pkl'))

        if os.path.exists(wf_name):
            print('{} already exists.'.format(wf_name))
        else:
            ris = []
            id = 1
            with open(rf_name, 'r', encoding='utf-8') as rf:
                for l in tqdm(rf.readlines(), desc='Buildiing trees from {}: '.format(f_name)):
                    ris_p, = parser.get_tree(l)
                    ris.append(ris_p)
                    id += 1

            with open(wf_name, 'wb') as wf:
                pkl.dump(ris, wf)


# the bin tree already parsed and dep_tree are not correctly aligned
def create_super_tree(const_t: nx.DiGraph, dep_t: nx.DiGraph):
    sort_fun = lambda x: x[1]['token_id']
    const_leaves = sorted([(i, d) for i, d in const_t.nodes(data=True) if 'token_id' in d], key=sort_fun)
    new_const_t = copy.deepcopy(const_t)

    rev_const_t = new_const_t.reverse(copy=False)
    n_wrong = 0
    for u, v, d in dep_t.edges(data=True):
        t = d['type']
        if v == 0:
            continue
        try:
            id_token_u = dep_t.nodes[u]['token_id']
            id_token_v = dep_t.nodes[v]['token_id']

            u_in_const = const_leaves[id_token_u - 1][0]
            v_in_const = const_leaves[id_token_v - 1][0]

            # get the tag
            new_const_t.add_node(u_in_const, tag=dep_t.nodes[u]['tag'])

            assert id_token_u == const_t.nodes[u_in_const]['token_id']
            assert id_token_v == const_t.nodes[v_in_const]['token_id']

            assert dep_t.nodes[u]['word'] == const_t.nodes[u_in_const]['word']
            assert dep_t.nodes[v]['word'] == const_t.nodes[v_in_const]['word']
            lca = nx.algorithms.lowest_common_ancestor(rev_const_t, u_in_const, v_in_const)
            assert lca is not None

            if 'type' in new_const_t.nodes[lca]:
                new_const_t.nodes[lca]['type'].append(t)
            else:
                new_const_t.add_node(lca, type=[t])
        except:
            n_wrong +=1

    return new_const_t, n_wrong


def bin_tree_to_nx(nltk_t, words_vocab):
    g = nx.DiGraph()
    token_id = 1

    def rec_parsing(node):
        nonlocal g
        nonlocal token_id

        my_id = g.number_of_nodes()
        if len(node) == 1 and isinstance(node[0], str):
            # leaf node
            w: str = node[0]
            w = w.replace('\/', '/')
            w.lower()
            if w in words_vocab:
                id_w = words_vocab[w]
            else:
                id_w = ConstValues.UNK

            g.add_node(my_id, x=id_w, mask=1, y=int(node.label()), token_id=token_id, )
            token_id += 1
        else:
            # internal node
            g.add_node(my_id, x=ConstValues.NO_ELEMENT, mask=0, y=int(node.label()))

            id_edge = 0
            for ch in node:
                ch_id = rec_parsing(ch)
                g.add_edge(ch_id, my_id, pos=id_edge)
                id_edge += 1

        return my_id

    rec_parsing(nltk_t)
    return g


def parse_bin_trees(input_file_path, words_vocab):
    ris = []
    with open(input_file_path, 'r', encoding='utf-8') as rf:
        for l in tqdm(rf.readlines(), desc='Reading trees from {}: '.format(os.path.basename(input_file_path))):
            t = string_to_nltk_tree(l)
            bin_t = bin_tree_to_nx(t, words_vocab)
            dgl_t = nx_to_dgl(bin_t, node_attrs=['x', 'y', 'mask'])
            ris.append(dgl_t)
    return ris


class SstPreprocessor:

    def __init__(self, config):
        self.config = config

    def preprocess(self):
        config = self.config
        input_dir = config.input_dir
        output_dir = config.output_dir
        file_names = {'train': config.training_file,
                      'validation': config.validation_file,
                      'test': config.test_file}
        words_vocab_file = config.words_vocabulary_file
        pretrained_embs_file = config.pretrained_embs_file
        embedding_dim = config.embedding_dim

        # load vocabulary
        words_vocab = {'unk': 0}
        with open(os.path.join(input_dir, words_vocab_file)) as f:
            for l in tqdm(f.readlines(), desc='Creating words vocabulary'):
                l = l.strip()
                words_vocab[l] = len(words_vocab)

        # preprocessing binary trees
        for tag_name, f in file_names.items():
            tree_list = parse_bin_trees(os.path.join(input_dir, f), words_vocab)
            with open(os.path.join(output_dir, '{}.pkl'.format(tag_name)), 'wb') as fw:
                pkl.dump(tree_list, fw)

        pretrained_embs = load_embeddings(pretrained_embs_file, words_vocab, embedding_dim=embedding_dim)
        with open(os.path.join(output_dir, 'pretrained_embs.pkl'), 'wb') as fw:
            pkl.dump(pretrained_embs, fw)