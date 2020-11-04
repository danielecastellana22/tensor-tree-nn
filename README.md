# Tensor Tree-RNN
We provide an extension of Tree Recursive Neural Network which is based on tensor theory.  

## Code Structure
The code structure is the following:
* `experiments` folder contains base code to run experiments;
* `models` folder contains implementation of both probabilistic and neural models; 
* `preprocessing` folder contains the code to preprocess dataset;
* `tasks` folder contains one folder for each tasks. Each folder contain utils function to preprocess and run experiments on the task;
* `utils` folder contains utils code.

## How to run an experiment
The folder `experiments` contains two files to run an experiment:
* `run.py --config-file c --num-worker n` which runs the experiment specified by the configuration file `c` using `n` process to parallelise the model selection;
* `preprocess.py --config-file c` which perform the preprocessing specified by the configuration file `c`.
## Config files

### Preprocessing 

### Neural Models

### Probabilistic Models