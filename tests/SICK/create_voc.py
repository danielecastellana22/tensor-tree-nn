import os
import re
from tqdm import tqdm


if __name__ == '__main__':
    mypath = 'data/sick'
    all_files = [os.path.join(mypath, f) for f in ['SICK_train_A.txt', 'SICK_trial_A.txt', 'SICK_test_A.txt',
                                                   'SICK_train_B.txt', 'SICK_trial_B.txt', 'SICK_test_B.txt']]

    vocab = {}
    with open(os.path.join(mypath, 'vocab.txt'), 'w') as f_voacb:
        for f_name in all_files:
            with open(f_name, 'r') as f:
                skip = True
                for l in tqdm(f.readlines()):
                    if not skip:
                        a_list = re.sub(' +', ' ', l.replace('(', '').replace(')', '').strip()).split(' ')
                        for w in a_list:
                            w = w.lower()
                            if w not in vocab:
                                vocab[w] = len(vocab)
                                f_voacb.write(w +' \n')
                    else:
                        skip = False
