import os
from abc import abstractmethod
from exputils.utils import eprint
from exputils.serialisation import to_json_file


class Preprocessor:

    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.dataset_stats = {}
        self.words_vocab = {}
        self.outputs_vocab = {}

    @abstractmethod
    def preprocess(self):
        raise NotImplementedError('This method must be implmented in a subclass!')

    def __init_stats__(self, tag_name):
        self.dataset_stats[tag_name] = {'tot_samples': 0}

    def __update_stats__(self, tag_name, t):

        # update dataset stats
        self.dataset_stats[tag_name]['tot_samples'] += 1

    def __print_stats__(self, tag_name):
        eprint('{} stats:'.format(tag_name))
        for k, v in self.dataset_stats[tag_name].items():
            eprint('{}:  {}.'.format(k, v))

    def __save_stats__(self):
        output_dir = self.output_dir

        # save all stats
        eprint('Saving dataset stats.')
        if len(self.words_vocab) > 0:
            to_json_file(self.words_vocab,os.path.join(output_dir, 'words_vocab.json'))
        if len(self.outputs_vocab) > 0:
            to_json_file(self.outputs_vocab, os.path.join(output_dir, 'outputs_vocab.json'))
        to_json_file(self.dataset_stats, os.path.join(output_dir, 'dataset_stats.json'))

    def __get_word_id__(self, w):
        return self.words_vocab.setdefault(w, len(self.words_vocab))

    def __get_output_id__(self, y):
        return self.outputs_vocab.setdefault(y, len(self.outputs_vocab))
