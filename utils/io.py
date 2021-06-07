import yaml
import pandas as pd
import os
import psutil
import torch

def load_yaml(path, key='parameters'):
    with open(path, 'r') as stream:
        try:
            return yaml.load(stream, Loader=yaml.FullLoader)[key]
        except yaml.YAMLError as exc:
            print(exc)

def save_dataframe_csv(df, path, name):
    df.to_csv(path + '/' + name, index=False)


def load_dataframe_csv(path, name=None, delimiter=None, names=None):
    if not name:
        return pd.read_csv(path, delimiter=delimiter, names=names)
    else:
        return pd.read_csv(path+name, delimiter=delimiter, names=names)

def check_ram_usage():
    """
    Compute the RAM usage of the current process.
        Returns:
            mem (float): Memory occupation in Megabytes
    """

    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024)

    return mem

def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state
