import argparse
from utils.io import load_yaml
from types import SimpleNamespace
from utils.utils import boolean_string
import time
import torch
import random
import numpy as np
from experiment.run import multiple_run

def main(args):
    genereal_params = load_yaml(args.general)
    data_params = load_yaml(args.data)
    agent_params = load_yaml(args.agent)
    genereal_params['verbose'] = args.verbose
    genereal_params['cuda'] = torch.cuda.is_available()
    final_params = SimpleNamespace(**genereal_params, **data_params, **agent_params)
    time_start = time.time()
    print(final_params)

    #reproduce
    np.random.seed(final_params.seed)
    random.seed(final_params.seed)
    torch.manual_seed(final_params.seed)
    if final_params.cuda:
        torch.cuda.manual_seed(final_params.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    #run
    multiple_run(final_params)



if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser('CVPR Continual Learning Challenge')
    parser.add_argument('--general', dest='general', default='config/general.yml')
    parser.add_argument('--data', dest='data', default='config/data/cifar100/cifar100_nc.yml')
    parser.add_argument('--agent', dest='agent', default='config/agent/er.yml')

    parser.add_argument('--verbose', type=boolean_string, default=True,
                        help='print information or not')
    args = parser.parse_args()
    main(args)