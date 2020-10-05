import os
from utils.misc import get_logger, create_datatime_dir
from utils.serialization import to_json_file, from_json_file, from_pkl_file, to_torch_file, from_torch_file
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
from experiments.base import Experiment


class ExperimentRunner:

    # TODO: add recovery strategy: a flag which indicates train, recover, test
    def __init__(self, output_dir, num_run, num_workers, metric_class_list, config_list, debug_mode):
        # self.experiment_class = experiment_class
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

        if n_config > 1:
            fs_list=[]
            for i_config, c in enumerate(self.config_list):
                for i_run in range(self.num_run):

                    exp_id = 'c{}_r{}'.format(i_config, i_run)
                    exp_out_dir = self.__get_conf_run_dir__(i_config, i_run)
                    os.makedirs(exp_out_dir)
                    output_msg = 'Configuration {} Run {} finished:'.format(i_config, i_run)

                    f = self.__start_single_exp__(c, exp_id, exp_out_dir, output_msg, do_test=False)
                    fs_list.append(f)

            if not self.debug_mode:
                self.logger.info('All configuration sumitted to the pool.')
                # this will wait the end of all subprocess
                concurrent.futures.wait(fs_list)

            self.logger.info('Model selection finished.')

            ms_validation_results = self.__load_all_validation_results__()
            to_json_file(ms_validation_results, os.path.join(self.output_dir, 'validation_results.json'))

            ms_validation_avg = np.mean(ms_validation_results[self.metric_class_list[0].get_name()], axis=1)

            if self.metric_class_list[0].HIGHER_BETTER:
                best_config_id = np.argmax(ms_validation_avg)
            else:
                best_config_id = np.argmin(ms_validation_avg)

            self.logger.info('Configuration {} is the best one! Validation Score: {}'.format(best_config_id, ms_validation_avg[best_config_id]))

            # save best config
            self.logger.info('Saving best configuration.')
            best_config = self.config_list[best_config_id]
            to_json_file(best_config, os.path.join(self.output_dir, 'best_config.json'))

            self.logger.info('Retraining and test the best configuration.')
        else:
            self.logger.info('There is only one configuration. No model selection is performed.')
            best_config = self.config_list[0]

        for i_run in range(self.num_run):
            self.logger.info('Testing Run {}.'.format(i_run))
            test_id = 'test{}'.format(i_run)
            test_out_dir = self.__get_test_run_dir__(i_run)
            os.makedirs(test_out_dir)
            output_msg = 'Testing run {} finished:'.format(i_run)

            self.__start_single_exp__(best_config, test_id, test_out_dir, output_msg, do_test=True)

        if not self.debug_mode:
            self.pool.shutdown()
        self.logger.info('Test finished.')

        # load all test results
        all_test_results = self.__load_all_test_results__()
        for k, v in all_test_results.items():
            self.logger.info('Test {}: {:4f} +/- {:4f}'.format(k, np.mean(v, axis=1)[0], np.std(v, axis=1)[0]))
        to_json_file(all_test_results, os.path.join(self.output_dir, 'test_results.json'))
        self.logger.info('Test results saved')

    @staticmethod
    def __exp_execution_fun__(exp_class, c, exp_id, exp_out_dir, metric_list, do_test, debug_mode):
        exp_logger = get_logger(exp_id, exp_out_dir, file_name='experiment.log', write_on_console=debug_mode)
        exp = exp_class(c, exp_out_dir, exp_logger, debug_mode)
        return exp.run_training(metric_list, do_test)

    def __start_single_exp__(self, config, exp_id, exp_out_dir, output_msg, do_test):

        def done_callback(fut):
            self.logger.info('{}: {}.'.format(output_msg, ' | '.join(map(str, fut.result()))))

        fun_params = [Experiment, config, exp_id, exp_out_dir, self.metric_class_list, do_test, self.debug_mode]
        if self.debug_mode:
            ris = self.__exp_execution_fun__(*fun_params)
            self.logger.info('{}: {}.'.format(output_msg, ' | '.join(map(str, ris))))
            return None
        else:
            f = self.pool.submit(self.__exp_execution_fun__, *fun_params)
            f.add_done_callback(done_callback)
            return f

    def __load_all_validation_results__(self):
        all_val_metrics = [[None for i in range(self.num_run)] for i in range(len(self.config_list))]
        for i_config in range(len(self.config_list)):
            for i_run in range(self.num_run):
                sub_dir = self.__get_conf_run_dir__(i_config, i_run)
                all_val_metrics[i_config][i_run] = from_json_file(os.path.join(sub_dir, 'best_validation_metrics.json'))

        return self.__metrics_list_to_val_dict__(all_val_metrics)

    def __load_all_test_results__(self):
        all_test_metrics = [[None for i in range(self.num_run)]]
        for i_run in range(self.num_run):
            sub_dir = self.__get_test_run_dir__(i_run)
            all_test_metrics[0][i_run] = from_json_file(os.path.join(sub_dir, 'test_metrics.json'))

        return self.__metrics_list_to_val_dict__(all_test_metrics)

    def __metrics_list_to_val_dict__(self, metrics_list):
        # metrics list MUST be a list of list
        val_dict = {}
        for c in self.metric_class_list:
            c_name = c.__name__
            val_dict[c_name] = [[y[c_name] for y in x] for x in metrics_list]
        return val_dict

    def __get_conf_run_dir__(self, id_config, n_run):
        return os.path.join(self.output_dir, 'conf_{}/run_{}'.format(id_config, n_run))

    def __get_test_run_dir__(self, n_run):
        return os.path.join(self.output_dir, 'test/run_{}'.format(n_run))
