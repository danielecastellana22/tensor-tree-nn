import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import torch as th

from exputils.serialisation import from_json_file
from exputils.utils import eprint, get_logger
from exputils.configurations import Config, ExpConfig
from exputils.experiments import Experiment


def __plot_matrix__(ax, cm, x_label, x_tick_label, y_label, y_tick_label, title=None, cmap='viridis', vmin=None, vmax=None, fmt='.2f'):
    if vmin is None:
        vmin = cm.min()

    if vmax is None:
        vmax = cm.max()

    cmap_obj = matplotlib.cm.get_cmap(cmap)

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap_obj, vmin=vmin, vmax=vmax)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=x_tick_label, yticklabels=y_tick_label,
           title=title,
           ylabel=y_label,
           xlabel=x_label)

    ax.set_xticks(np.arange(cm.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0]+1)-.5, minor=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.

    def get_text_color(v):
        v = (v-vmin)/(vmax-vmin)
        rgba = cmap_obj(v)
        lum=0.2126*rgba[0] + 0.7152*rgba[1] + 0.0722*rgba[2]
        if lum<0.5:
            return 'white'
        else:
            return 'black'

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color=get_text_color(cm[i, j]))

    return ax


def plot_confusion_matrix(ax, cm, classes_name=None, title=None):
    #cm = cm / cm.sum()
    if classes_name == None:
        classes_name = list(range(cm.shape[0]))
    __plot_matrix__(ax, cm,
                    x_label='Predicted label', x_tick_label=classes_name, fmt='d',
                    y_label='True label', y_tick_label=classes_name, title=title, cmap='Blues')


def __get_run_exp_dir_and_config_path__(model_dir, run_exp_dir=None):
    if run_exp_dir is not None:
        results_dir = os.path.join(model_dir, run_exp_dir)
    else:
        ls_dir = sorted([x for x in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, x)) and x.startswith('2')])
        if len(ls_dir) == 0:
            eprint('WARNING! {} does not contain results folder!'.format(model_dir))
            return
        else:
            if len(ls_dir) > 1:
                eprint('WARNING! There are mroe than one folder in {}. We select the most recent!'.format(model_dir))
            results_dir = os.path.join(model_dir, ls_dir[-1])

    config_exp_path = os.path.join(model_dir, 'config.yaml')

    return results_dir, config_exp_path


def read_ms_results(model_dir, run_exp_dir=None):

    results_dir, config_exp_path = __get_run_exp_dir_and_config_path__(model_dir, run_exp_dir)

    out = {}
    if os.path.exists(os.path.join(results_dir, 'test_results.json')):
        test_results = from_json_file(os.path.join(results_dir, 'test_results.json'))
    else:
        eprint('WARNING! {} is missing! try to recover it in a test folder'.format(os.path.join(results_dir, 'test_results.json')))
        p = os.path.join(results_dir, '../test')
        if os.path.exists(p):
            eprint('test folder exists! try to read an exp folder in it')
            p = os.path.join(p, os.listdir(p)[0])
            test_results = from_json_file(os.path.join(p, 'test_results.json'))
        else:
            eprint(''.format(p))
            raise FileNotFoundError('test_results.json not found')

    # trannsofrm test results
    out['test_results'] = {}
    for k in test_results:
        out['test_results'][k] = np.array(test_results[k])

    n_params_dict = __read_same_json_from_ms_folders__(results_dir, 'num_model_parameters.json')
    info_tr_dict = __read_same_json_from_ms_folders__(results_dir, 'info_training.json')

    if os.path.exists(os.path.join(results_dir, 'validation_results.json')):
        validation_results = from_json_file(os.path.join(results_dir, 'validation_results.json'))
    else:
        eprint('WARNING! {} is missing! try to recover it'.format(os.path.join(results_dir, 'validation_results.json')))
        validation_results = __read_same_json_from_ms_folders__(results_dir, 'best_validation_metrics.json')

    # get dict grid
    grid_dict, n_run = ExpConfig.get_grid_dict(config_exp_path)
    reshape_size = [len(x) for x in grid_dict.values()]
    reshape_size.append(n_run)
    out['validation_results'] = {}
    first_val_results = None
    for i, k in enumerate(validation_results):
        if i == 0:
            first_val_results = np.array(validation_results[k]).reshape(*reshape_size)
        out['validation_results'][k] = np.array(validation_results[k]).reshape(*reshape_size)

    out['num_params'] = {k: np.array(v).reshape(*reshape_size) for k, v in n_params_dict.items()}
    out['info_training'] = {k: np.array(v).reshape(*reshape_size) if len(v) == np.prod(reshape_size) else v for k, v in info_tr_dict.items()}
    out['id_best_config'] = np.argmax(np.mean(first_val_results.reshape(-1, first_val_results.shape[-1]), axis=-1),)
    out['params_grid'] = grid_dict

    return out


def __read_same_json_from_ms_folders__(result_dir, file_name):
    out_array = []
    id_conf = 0
    conf_dir = os.path.join(result_dir, 'conf_{}'.format(id_conf))
    while os.path.exists(conf_dir):
        out_array.append([])
        id_run = 0
        run_dir = os.path.join(conf_dir, 'run_{}'.format(id_run))
        while os.path.exists(run_dir):
            if os.path.exists(os.path.join(run_dir, file_name)):
                ris = from_json_file(os.path.join(run_dir, file_name))
            else:
                eprint('WARNING! {} is missing!'.format(os.path.join(run_dir, file_name)))
                ris = None

            out_array[id_conf].append(ris)
            id_run += 1
            run_dir = os.path.join(conf_dir, 'run_{}'.format(id_run))

        id_conf += 1
        conf_dir = os.path.join(result_dir, 'conf_{}'.format(id_conf))

    out_d = {}
    for k in out_array[0][0]:
        out_d[k] = []
        for y in out_array:
            out_d[k].append([])
            for x in y:
                if x is not None:
                    out_d[k][-1].append(x[k])
                else:
                    out_d[k][-1].append(np.nan)
    return out_d


def get_exp_best_model_best_pred(model_dir, out_dir, run_exp_dir=None, id_run=-1):

    results_dir, config_exp_path = __get_run_exp_dir_and_config_path__(model_dir, run_exp_dir)
    eprint('LOADING {}'.format(results_dir))
    name = 'test'
    m_logger = get_logger(name, out_dir, '{}.log'.format(name), True)
    if os.path.exists(os.path.join(results_dir, 'best_config.json')):
        m_best_config = Config.from_json_fle(os.path.join(results_dir, 'best_config.json'))
    else:
        # probably no model selection has been performed
        m_best_config = Config.from_yaml_file(os.path.join(results_dir, '../config.yaml'))
    m_exp = Experiment(config=m_best_config, output_dir=out_dir, logger=m_logger, debug_mode=True)
    if id_run == -1:
        best_test_id = np.argmax(list(from_json_file(os.path.join(results_dir, 'test_results.json')).values())[0])
    else:
        best_test_id = id_run
    m = m_exp.__create_exp_module__()
    m.load_state_dict(th.load(os.path.join(results_dir, 'test/run_{}/params_learned.pth'.format(best_test_id))))

    pred = th.load(os.path.join(results_dir, 'test/run_{}/test_prediction.pth'.format(best_test_id)))
    return m_exp, m, pred