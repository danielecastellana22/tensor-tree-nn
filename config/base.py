from utils.utils import string2class
from utils.serialization import from_yaml_file, from_json_file
import copy


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
            super(Config, self).__getattr__(item)

    @classmethod
    def from_json_fle(cls, path):
        return cls(**from_json_file(path))

    @classmethod
    def from_yaml_file(cls, path):
        return cls(**from_yaml_file(path))


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
        config_dict = from_yaml_file(path)
        exp_runner_params = config_dict.pop('experiment_config')
        exp_runner_params['experiment_class'] = string2class(exp_runner_params['experiment_class'])
        exp_runner_params['metric_class_list'] = list(map(string2class, exp_runner_params['metric_class_list']))

        config_dict_list = ExpConfig.__build_grid_search__(config_dict)
        ris = []
        for d in config_dict_list:
            ris.append(Config(**d))

        return exp_runner_params, ris
