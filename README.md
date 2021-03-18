# Tensor Recursive Model
We provide an extension neural and probabilistic models of recursive models for tree-structure data based on tensor theory.

**List of publication based on this repository:**

- ["**Generalising Recursive Neural Models by Tensor Decomposition**"](http://pages.di.unipi.it/castellana/publication/ijcnn20/),<br>
[Daniele Castellana](http://pages.di.unipi.it/castellana/), [Davide Bacciu](http://pages.di.unipi.it/bacciu/),<br>
 International Joint Conference on Neural Networks (IJCNN), Glasgow, UK, 2020  

- ["**Tensor Decompositions in Recursive Neural Networks for Tree-Structured Data**"](http://pages.di.unipi.it/castellana/publication/esann20/),<br>
[Daniele Castellana](http://pages.di.unipi.it/castellana/), [Davide Bacciu](http://pages.di.unipi.it/bacciu/),<br>
European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, Bruges, Belgium,  2020
 
 - ["**Learning from Non-Binary Constituency Trees via Tensor Decomposition**"](http://pages.di.unipi.it/castellana/publication/coling20/),<br>
[Daniele Castellana](http://pages.di.unipi.it/castellana/), [Davide Bacciu](http://pages.di.unipi.it/bacciu/),<br>
Proceedings of the 28th International Conference on Computational Linguistics, Barcelona, Spain, 2020

## Code Structure

The code is structured as follows:
- `data` contains the BoolSent dataset
- `models` contains the implementation of the recursive models (both probabilistic and neural); 
- `preprocessing` contains the code to preprocess the datasets;
- `tasks` contains all the files to execute the experiments; there is a readme files for each task;
- `tree_utils` contains utils code.

Also, two other repositories are used:
- `exputils` provides the utils to run an experiment, train a model, parse configuration, etc..
- `thlogprob` provides a minimal library to handle distribution using pytorch.

See the readme for more information.

## How to run an experiment

1) download the raw data (see [next section](#where-to-download-the-dataset));
2) in NLP tasks, sentences should be parsed running the command:

    `python tasks/task_name/parse_raw_data raw_data_folder output_folder`
3) Run the preprocessor using the command:

    `python preprocess.py --config-file preproc_config_file`,<br>
    where `preproc_config_file` can be found in the folder `tasks/task_name/config_files`
4) Run the experiment using the command:

    `python run.py --config-file run_config_file`,<br>
    where `run_config_file` can be found in the folder `tasks/task_name/config_files`.


#### Where to download the dataset

- [*ListOps*](https://github.com/nyu-mll/spinn/tree/listops-release/python/spinn/data/listops)
- [*LRT*](https://github.com/sleepinyourhat/vector-entailment/tree/master/propositionallogic)
- [*SICK*](https://alt.qcri.org/semeval2014/task1/index.php?id=data-and-tools)
- [*SST*](https://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip)
- [*TREC*](https://cogcomp.seas.upenn.edu/Data/QA/QC/)