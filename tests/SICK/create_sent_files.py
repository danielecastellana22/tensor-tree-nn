import os
from tqdm import tqdm
from nltk.parse.corenlp import CoreNLPDependencyParser


if __name__ == '__main__':
    mypath = 'data/sick'
    all_files = [os.path.join(mypath, f) for f in ['SICK_train.txt', 'SICK_trial.txt', 'SICK_test.txt']]

    dep_parser = CoreNLPDependencyParser(url='http://localhost:9000')
    for f_name in all_files:
        with open(f_name, 'r') as f, \
             open(f_name.replace('.txt', '_ID.txt'), 'w') as f_ID, \
             open(f_name.replace('.txt', '_A.txt'), 'w') as f_A,\
             open(f_name.replace('.txt', '_B.txt'), 'w') as f_B,\
             open(f_name.replace('.txt', '_SCORE.txt'), 'w') as f_SCORE, \
             open(f_name.replace('.txt', '_JUD.txt'), 'w') as f_JUD:
            skip = True
            for l in tqdm(f.readlines()):
                if not skip:
                    v = l.split('\t')
                    f_ID.write(v[0] + '\n')
                    a, = dep_parser.raw_parse(v[1])
                    b, = dep_parser.raw_parse(v[2])
                    f_A.write(str(a.tree()).replace('\n', '').replace('n\'t','not') + '\n')
                    f_B.write(str(b.tree()).replace('\n', '').replace('n\'t','not') + '\n')
                    f_SCORE.write(v[3] + '\n')
                    f_JUD.write(v[4])
                else:
                    skip = False