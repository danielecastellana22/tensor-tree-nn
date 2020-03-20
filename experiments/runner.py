import os
from utils.utils import get_logger, create_datatime_dir
from utils.serialization import to_json_file, from_json_file, from_pkl_file, to_torch_file, from_torch_file
import numpy as np
from concurrent.futures import ProcessPoolExecutor


class ExperimentRunner:

    # TODO: add recovery strategy
    def __init__(self, experiment_class, output_dir, num_run, num_workers, metric_class_list, config_list, debug_mode=False):
        self.experiment_class = experiment_class
        self.config_list = config_list
        self.num_run = num_run
        self.output_dir = create_datatime_dir(output_dir)
        self.logger = get_logger('runner', self.output_dir, file_name='runner.log', write_on_console=True)
        self.metric_class_list = metric_class_list
        self.debug_mode = debug_mode
        if not self.debug_mode:
            self.pool = ProcessPoolExecutor(max_workers=num_workers)

    def run(self):

        n_config = len(self.config_list)
        self.logger.info('Model selection starts: {} configurations to run {} times.'.format(n_config, self.num_run))

        for i_config, c in enumerate(self.config_list):
            for i_run in range(self.num_run):
                self.logger.info('Configuration {} Run {} launched.'.format(i_config, i_run))

                exp_id = 'c{}_r{}'.format(i_config, i_run)
                exp_out_dir = self.__get_conf_run_folder__(i_config, i_run)
                os.makedirs(exp_out_dir)

                exp_logger = get_logger(exp_id, exp_out_dir, file_name='training.log', write_on_console=False)
                exp = self.experiment_class(c, exp_out_dir, exp_logger)
                if self.debug_mode:
                    ris = exp.run_training(self.metric_class_list)
                    self.logger.info('Configuration {} Run {} finished: {}.'.format(i_config, i_run, ' | '.join(map(str, ris))))
                else:
                    f = self.pool.submit(exp.run_training, self.metric_class_list)
                    f.add_done_callback(self.__get_done_callback__(i_config, i_run))

        self.pool.shutdown()  # this will wai the end of all subprocess
        self.logger.info('Model selection finished.')

        ms_dev_results = self.__load_all_dev_results__()
        to_json_file(ms_dev_results, os.path.join(self.output_dir, 'all_validation_results.json'))

        ms_dev_mean = np.mean(ms_dev_results[self.metric_class_list[0].__name__], axis=1)

        if self.metric_class_list[0].HIGHER_BETTER:
            best_config_id = np.argmax(ms_dev_mean)
        else:
            best_config_id = np.argmin(ms_dev_mean)

        self.logger.info('Configuration {} is the best one! Validation Score: {}'.format(best_config_id, ms_dev_mean[best_config_id]))

        # save best config
        self.logger.info('Saving best configuration.')
        to_json_file(self.config_list[best_config_id], os.path.join(self.output_dir, 'best_config.json'))

        # load best weight from the first run
        self.logger.info('Load best model weight.')
        self.logger.info('Testing the best configuration')
        all_test_results = [None] * self.num_run
        for i_run in range(self.num_run):
            best_model_weight = self.__get_model_weight__(best_config_id, i_run)

            self.logger.info('Testing Run {}.'.format(i_run))
            test_exp = self.experiment_class(self.config_list[best_config_id], self.output_dir,
                                             self.logger.getChild('best_model_testing'))

            test_metrics, test_prediction = test_exp.run_test(best_model_weight, self.metric_class_list)
            all_test_results[i_run] = {type(x).__name__: x.get_value() for x in test_metrics}
            self.logger.info('Test {} results: {}.'.format(i_run, ' | '.join(map(str, test_metrics))))

        self.logger.info('Save test results.')
        to_json_file(self.__metrics_list_to_val_dict__(all_test_results), os.path.join(self.output_dir, 'test_results.json'))

    def __get_done_callback__(self, i_config, i_run):
        def f(fut):
            self.logger.info('Configuration {} Run {} finished: {}.'.format(i_config, i_run, ' | '.join(map(str, fut.result()))))

        return f

    def __load_all_dev_results__(self):
        all_dev_metrics = [[None] * self.num_run] * len(self.config_list)
        for i_config in range(len(self.config_list)):
            for i_run in range(self.num_run):
                sub_dir = self.__get_conf_run_folder__(i_config, i_run)
                all_dev_metrics[i_config][i_run] = from_json_file(os.path.join(sub_dir, 'best_dev_metrics.json'))

        return self.__metrics_list_to_val_dict__(all_dev_metrics)

    def __metrics_list_to_val_dict__(self, metrics_list):

        def __rec_conv__(l, c):
            if isinstance(l, list):
                return [__rec_conv__(x, c) for x in l]
            else:
                return l[c]

        val_dict = {}
        for c in self.metric_class_list:
            c_name = c.__name__
            val_dict[c_name] = __rec_conv__(metrics_list, c_name)
        return val_dict

    def __get_model_weight__(self, id_config, n_run):
        sub_dir = self.__get_conf_run_folder__(id_config, n_run)
        return from_torch_file(os.path.join(sub_dir, 'model_weight.pth'))

    def __get_conf_run_folder__(self, id_config, n_run):
        return os.path.join(self.output_dir, 'conf_{}/run_{}'.format(id_config, n_run))
