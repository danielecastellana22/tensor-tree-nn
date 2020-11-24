from utils.misc import string2class
from utils.serialization import from_yaml_file, from_json_file
import copy
from collections import OrderedDict


class Config(dict):

    def __init__(self, **config_dict):
        # store jsno representation
        super(Config, self).__init__()

        # set attributes
        for k, v in config_dict.items():
            if isinstance(v, dict):
                # if is dict, create a new Config obj
                v = Config(**v)
            super(Config, self).__setitem__(k, v)

    # the dot works as []
    def __getattr__(self, item):
        if item in self:
            return self.__getitem__(item)
        else:
            raise AttributeError('The key {} must be specified!'.format(item))

    @classmethod
    def from_json_fle(cls, path):
        return cls(**from_json_file(path))

    @classmethod
    def from_yaml_file(cls, path):
        return cls(**from_yaml_file(path))


class ExpConfig:

    @staticmethod
    def __build_config_list__(config_dict):

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
                    out_list += __rec_build__(d, k_list[1:], d_out)
            else:
                d_out[k] = v
                out_list += __rec_build__(d, k_list[1:], d_out)

            return out_list

        return __rec_build__(config_dict, list(config_dict.keys()), {})

    @staticmethod
    def from_file(path):
        config_dict = from_yaml_file(path)
        exp_runner_params = config_dict.pop('experiment_config')
        if 'experiment_class' in exp_runner_params:
            raise ValueError('Old config file!')


        exp_runner_params['metric_class_list'] = list(map(string2class, exp_runner_params['metric_class_list']))

        config_list = ExpConfig.__build_config_list__(config_dict)

        ris = []
        for d in config_list:
            ris.append(Config(**d))

        return exp_runner_params, ris

    @staticmethod
    def __build_grid_dict__(config_dict):
        d_out = OrderedDict()

        def __rec_build__(d, k_pre):
            for k, v in d.items():
                if isinstance(v, list):
                    d_out[k_pre+'.'+k] = copy.deepcopy(v)
                elif isinstance(v, dict):
                    __rec_build__(v, k)

        __rec_build__(config_dict, '')
        return d_out

    @staticmethod
    def get_grid_dict(path):
        config_dict = from_yaml_file(path)
        exp_config = config_dict.pop('experiment_config')
        if 'experiment_class' in exp_config:
            raise ValueError('Old config file!')
        return ExpConfig.__build_grid_dict__(config_dict), exp_config['num_run']


def create_object_from_config(obj_config, **other_params):
    class_name = string2class(obj_config['class'])
    params = obj_config['params'] if 'params' in obj_config else {}
    params.update(other_params)
    return class_name(**params)