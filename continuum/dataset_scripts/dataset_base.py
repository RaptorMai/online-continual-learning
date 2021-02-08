from abc import ABC, abstractmethod
import os

class DatasetBase(ABC):
    def __init__(self, dataset, scenario, task_nums, run, params):
        super(DatasetBase, self).__init__()
        self.params = params
        self.scenario = scenario
        self.dataset = dataset
        self.task_nums = task_nums
        self.run = run
        self.root = os.path.join('./datasets', self.dataset)
        self.test_set = []
        self.val_set = []
        self._is_properly_setup()
        self.download_load()


    @abstractmethod
    def download_load(self):
        pass

    @abstractmethod
    def setup(self, **kwargs):
        pass

    @abstractmethod
    def new_task(self, cur_task, **kwargs):
        pass

    def _is_properly_setup(self):
        pass

    @abstractmethod
    def new_run(self, **kwargs):
        pass

    @property
    def dataset_info(self):
        return self.dataset

    def get_test_set(self):
        return self.test_set

    def clean_mem_test_set(self):
        self.test_set = None
        self.test_data = None
        self.test_label = None