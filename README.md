#Tensor Tree-RNN
We provide an extension of Tree Recursive Nerual Network which is based on tensor theory.  

##Code Structure
The code structure is the following:
* `cannon` folder contains auxilary file for the hyper-parameter model selection;
* `data` folder contains data for the experiments; 
* `experiments` folder contains the code to run experiments;
* `treeRNN` folder contains the code of the Tenrsor Tree Recursive Model.

## How to run an experiment
The folder `experiments` contains two files to run an experiment:
* `single_run.py` which trains and tests a model on a given dataset;
* `model_selection.py` which executes a grid search of model on a given dataset.

**These to file are the only scripts which must be executed**. All the other information necessary to run an experiment 
(e.g. the model, the configuration, the dataset) are specified as parameters.

In the following, we show the most useful execution parameters.
* Experiment parameters:
    * `--dataset d`: the name of the dataset. `d` must be in `{lrt, list_ops, toy_bool_2, toy_bool_3, toy_bool_4, toy_bool_5}`;
    * `--tree-model m`: the name of the model. `d` must be in `{treeRNN, treeLSTM}`;
    * `--cell-type c`: the type of the aggregation. `c` must be in `{sumchild, full, hosvd, canonical, tt}`;
* Learning parameters:
    * `--gpu n`: the GPU id. If `n` is negative, it indicates the number of CPU cores; 
    * `--batch-size b`: the training batch size;
    * `--early-stopping a`: the number of epoch for the training early stopping;
    * `--epochs e`: the maximum number of training epochs;
    * `--weight-decay wd`: the L2 penalisation.
* Model parameters:
    * `--x-size ms`: the size of visible labels;
    * `--h-size hs`: the size of hidden states;
    * `--others s`: a JSON string which can contain other parameters (e.g. the tensor decomposition rank).
    
For example, the command:

`python single_run.py --gpu -2 --tree-model treeLSTM --cell-type canonical --dataset toy_bool_5 --h-size 50 --weight-decay 0.001 --others "{'rank': 3,'use_one_hot': 1}" --early-stopping 5`
    
train a Tree-LSTM model with the canonical decomposition as aggregation function on the boolean syntetic dataset with outdegree 5.

