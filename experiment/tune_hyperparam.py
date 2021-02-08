from types import SimpleNamespace
from sklearn.model_selection import ParameterGrid
from utils.setup_elements import setup_opt, setup_architecture
from utils.utils import maybe_cuda
from utils.name_match import agents
import numpy as np
from experiment.metrics import compute_performance


def tune_hyper(tune_data, tune_test_loaders, default_params, tune_params):
    param_grid_list = list(ParameterGrid(tune_params))
    print(len(param_grid_list))
    tune_accs = []
    tune_fgt = []
    for param_set in param_grid_list:
        final_params = vars(default_params)
        print(param_set)
        final_params.update(param_set)
        final_params = SimpleNamespace(**final_params)
        accuracy_list = []
        for run in range(final_params.num_runs_val):
            tmp_acc = []
            model = setup_architecture(final_params)
            model = maybe_cuda(model, final_params.cuda)
            opt = setup_opt(final_params.optimizer, model, final_params.learning_rate, final_params.weight_decay)
            agent = agents[final_params.agent](model, opt, final_params)
            for i, (x_train, y_train, labels) in enumerate(tune_data):
                print("-----------tune run {} task {}-------------".format(run, i))
                print('size: {}, {}'.format(x_train.shape, y_train.shape))
                agent.train_learner(x_train, y_train)
                acc_array = agent.evaluate(tune_test_loaders)
                tmp_acc.append(acc_array)
            print(
                "-----------tune run {}-----------avg_end_acc {}-----------".format(run, np.mean(tmp_acc[-1])))
            accuracy_list.append(np.array(tmp_acc))
        accuracy_list = np.array(accuracy_list)
        avg_end_acc, avg_end_fgt, avg_acc, avg_bwtp, avg_fwt = compute_performance(accuracy_list)
        tune_accs.append(avg_end_acc[0])
        tune_fgt.append(avg_end_fgt[0])
    best_tune = param_grid_list[tune_accs.index(max(tune_accs))]
    return best_tune