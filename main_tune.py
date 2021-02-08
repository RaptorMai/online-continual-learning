import argparse
from utils.io import load_yaml
from types import SimpleNamespace
from utils.utils import boolean_string
import time
import torch
import random
import numpy as np
from experiment.run import multiple_run_tune_separate
from utils.setup_elements import default_trick

def main(args):
    genereal_params = load_yaml(args.general)
    data_params = load_yaml(args.data)
    default_params = load_yaml(args.default)
    tune_params = load_yaml(args.tune)
    genereal_params['verbose'] = args.verbose
    genereal_params['cuda'] = torch.cuda.is_available()
    genereal_params['train_val'] = args.train_val
    if args.trick:
        default_trick[args.trick] = True
    genereal_params['trick'] = default_trick
    final_default_params = SimpleNamespace(**genereal_params, **data_params, **default_params)

    time_start = time.time()
    print(final_default_params)
    print()

    #reproduce
    np.random.seed(final_default_params.seed)
    random.seed(final_default_params.seed)
    torch.manual_seed(final_default_params.seed)
    if final_default_params.cuda:
        torch.cuda.manual_seed(final_default_params.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    #run
    multiple_run_tune_separate(final_default_params, tune_params, args.save_path)



if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser('Continual Learning')
    parser.add_argument('--general', dest='general', default='config/general_1.yml')
    parser.add_argument('--data', dest='data', default='config/data/cifar100/cifar100_nc.yml')
    parser.add_argument('--default', dest='default', default='config/agent/er/er_1k.yml')
    parser.add_argument('--tune', dest='tune', default='config/agent/er/er_tune.yml')
    parser.add_argument('--save-path', dest='save_path', default=None)
    parser.add_argument('--verbose', type=boolean_string, default=False,
                        help='print information or not')
    parser.add_argument('--train_val', type=boolean_string, default=False,
                        help='use tha val batches to train')
    parser.add_argument('--trick', type=str, default=None)
    args = parser.parse_args()
    main(args)