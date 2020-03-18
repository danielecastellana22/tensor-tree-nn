import yaml
from utils.utils import string2class
import copy
import json


class Config(dict):

    def __init__(self, **config_dict):
        # store jsno representation
        super(Config, self).__init__()
        self.__dict__['__json_repr__'] = json.dumps(config_dict, indent='\t')

        # set attributes
        for k, v in config_dict.items():
            if isinstance(v, dict):
                # if is dict, create a new Config obj
                v = Config(**v)
            else:
                # check if the string should be converted in class
                if k.endswith('_class'):
                    if isinstance(v, str):
                        v = string2class(v)
                    else:
                        self.string_list2class_list(v)
            super(Config, self).__setitem__(k, v)

    # the dot works as []
    def __getattr__(self, item):
        return self.__getitem__(item)

    # the config is immutable
    def __setattr__(self, key, value):
        raise AttributeError('The Config class cannot be modified!')

    def __setitem__(self, key, value):
        raise AttributeError('The Config class cannot be modified!')

    def __delitem__(self, key):
        raise AttributeError('The Config class cannot be modified!')

    @staticmethod
    def string_list2class_list(val_to_convert):

        def __rec_apply_list__(l):
            for i in range(len(l)):
                if isinstance(l[i], str):
                    l[i] = string2class(l[i])
                elif isinstance(l[i], list):
                    __rec_apply_list__(l[i])

        __rec_apply_list__(val_to_convert)

    def to_json(self):
        return self.__json_repr__

    @classmethod
    def from_json(cls, json_string):
        return cls(**json.loads(json_string))

    @classmethod
    def from_file(cls, path):
        config_dict = yaml.load(open(path, 'r'), Loader=yaml.SafeLoader)
        return cls(**config_dict)


class ExpConfig:

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

    @staticmethod
    def from_file(path):
        config_dict = yaml.load(open(path, 'r'), Loader=yaml.SafeLoader)
        exp_config = config_dict.pop('experiment_config')
        exp_config['experiment_class'] = string2class(exp_config['experiment_class'])

        config_dict_list = ExpConfig.__build_grid_search__(config_dict)
        ris = []
        for d in config_dict_list:
            ris.append(Config(**d))

        return exp_config, ris
