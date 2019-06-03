import os
import numpy as np
from nltk.parse import stanford

if __name__ == '__main__':
    mypath = 'data/sick'
    'C:/Users/Daniele Castellana/PycharmProjects/stanford-english-corenlp-2018-10-05-models.jar'
    os.environ['STANFORD_PARSER'] = 'C:/Users/Daniele Castellana/OneDrive - University of Pisa/project/matlab/datasets/utils/NLP-parser/stanford-parser-full-2018-10-17/stanford-parser.jar'
    os.environ['STANFORD_MODELS'] = 'C:/Users/Daniele Castellana/OneDrive - University of Pisa/project/matlab/datasets/utils/NLP-parser/stanford-parser-full-2018-10-17/stanford-parser-3.9.2-models.jar'
    all_files = [os.path.join(mypath,f) for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f)) and f.startswith('SICK')]

    p = stanford.StanfordNeuralDependencyParser()
    for f_name in all_files:
        with open(f_name, 'r') as f, \
             open(f_name.replace('.txt', '_ID.txt'), 'w') as f_ID, \
             open(f_name.replace('.txt', '_A.txt'), 'w') as f_A,\
             open(f_name.replace('.txt', '_B.txt'), 'w') as f_B,\
             open(f_name.replace('.txt', '_SCORE.txt'), 'w') as f_SCORE, \
             open(f_name.replace('.txt', '_JUD.txt'), 'w') as f_JUD:
            skip = True
            for l in f.readlines():
                if not skip:
                    v = l.split('\t')
                    f_ID.write(v[0] + '\n')
                    a = p.raw_parse(v[1])
                    f_A.write(v[1] + '\n')
                    f_B.write(v[2] + '\n')
                    f_SCORE.write(v[3] + '\n')
                    f_JUD.write(v[4])
                else:
                    skip = False