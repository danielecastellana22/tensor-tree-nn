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