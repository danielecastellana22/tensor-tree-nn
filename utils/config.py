import yaml
from utils.utils import string2class
import copy
import json

class Config:

    # TODO: use yaml schema
    # MANDATORY_FIELDS = ['experiment_class', 'training_config', 'tree_model_config']
    # TRAINING_CONFIG_FIELDS = ['device', 'batch_size', 'early_stopping_patience', 'metric_class', 'n_epochs']
    # TREE_MODEL_CONFIG_FIELDS = ['cell_class', 'aggregator_class', 'x_size', 'h_size',
    #                             'pos_stationairty', 'weight_decay']

    def __init__(self, config_dict):
        self.string_repr = json.dumps(config_dict, indent='\t')
        # convert string to class
        self.convert_string2class(config_dict)
        for k, v in config_dict.items():
            setattr(self, k, v)

    def __str__(self):
        return self.string_repr

    @staticmethod
    def convert_string2class(conf_dict):
        # We do not ammit list of dict
        def __rec_apply_list__(l):
            for i in range(len(l)):
                if isinstance(l[i], str):
                    l[i] = string2class(l[i])
                elif isinstance(l[i], list):
                    __rec_apply_list__(l[i])

        def __rec_visit_dict__(d):
            for k, v in d.items():
                if 'class' in k:
                    if isinstance(v, str):
                        d[k] = string2class(v)
                    elif isinstance(v, list):
                        __rec_apply_list__(v)
                else:
                    if isinstance(v, dict):
                        __rec_visit_dict__(v)

        __rec_visit_dict__(conf_dict)

    @staticmethod
    def __build_grid_search__(config_dict):

        def __rec_build__(d, k_list, d_out):
            if len(k_list) == 0:
                return [copy.deepcopy(d_out)]
            out_list = []
            k = k_list[0]
            v = d[k]
            if isinstance(v, dict):
                # now becomes a list
                v = __rec_build__(v, list(v.keys()), {})

            if isinstance(v, list):
                for vv in v:
                    d_out[k] = vv
                    out_list += __rec_build__(d,k_list[1:], d_out)
            else:
                d_out[k] = v
                out_list += __rec_build__(d, k_list[1:], d_out)

            return out_list

        return __rec_build__(config_dict, list(config_dict.keys()), {})

    @classmethod
    def from_file(cls, path):
        config_dict = yaml.load(open(path, 'r'), Loader=yaml.SafeLoader)
        exp_config = config_dict.pop('experiment_config')
        exp_config['experiment_class'] = string2class(exp_config['experiment_class'])

        config_dict_list = cls.__build_grid_search__(config_dict)
        ris = []
        for d in config_dict_list:
            ris.append(cls(d))

        return exp_config, ris

    @classmethod
    def from_json(cls, json_string):
        return cls(json.loads(json_string))
