import networkx as nx
import dgl
import dgl.backend as F
from nltk.corpus.reader import BracketParseCorpusReader


class TreeDataset(object):

    def __init__(self, file_name):
        self.file_name = file_name

    def _load(self):
        files = ['{}.txt'.format(self.mode)]
        corpus = BracketParseCorpusReader('{}/sst'.format(self.dir), files)
        sents = corpus.parsed_sents(files[0])
        # build trees
        for sent in sents:
            self.trees.append(self._build_tree(sent))

    def _build_tree(self, root):
        g = nx.DiGraph()
        def _rec_build(nid, node):
            for child in node:
                cid = g.number_of_nodes()
                if isinstance(child[0], str) or isinstance(child[0], bytes):
                    # leaf node
                    word = self.vocab.get(child[0].lower(), self.UNK_WORD)
                    g.add_node(cid, x=word, y=int(child.label()), mask=1)
                else:
                    g.add_node(cid, x=SST.PAD_WORD, y=int(child.label()), mask=0)
                    _rec_build(cid, child)
                g.add_edge(cid, nid)
        # add root
        g.add_node(0, x=SST.PAD_WORD, y=int(root.label()), mask=0)
        _rec_build(0, root)
        ret = dgl.DGLGraph()
        ret.from_networkx(g, node_attrs=['x', 'y', 'mask'])
        return ret

    def __getitem__(self, idx):
        """Get the tree with index idx.

        Parameters
        ----------
        idx : int
            Tree index.

        Returns
        -------
        dgl.DGLGraph
            Tree.
        """
        return self.trees[idx]

    def __len__(self):
        """Get the number of trees in the dataset.

        Returns
        -------
        int
            Number of trees.
        """
        return len(self.trees)