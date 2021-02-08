import glob
from PIL import Image
import numpy as np
from continuum.dataset_scripts.dataset_base import DatasetBase
import time
from continuum.data_utils import shuffle_data


class OpenLORIS(DatasetBase):
    """
    tasks_nums is predefined and it depends on the ns_type.
    """
    def __init__(self, scenario, params):  # scenario refers to "ni" or "nc"
        dataset = 'openloris'
        self.ns_type = params.ns_type
        task_nums = openloris_ntask[self.ns_type]  # ns_type can be (illumination, occlusion, pixel, clutter, sequence)
        super(OpenLORIS, self).__init__(dataset, scenario, task_nums, params.num_runs, params)


    def download_load(self):
        s = time.time()
        self.train_set = []
        for batch_num in range(1, self.task_nums+1):
            train_x = []
            train_y = []
            test_x = []
            test_y = []
            for i in range(len(datapath)):
                train_temp = glob.glob('datasets/openloris/' + self.ns_type + '/train/task{}/{}/*.jpg'.format(batch_num, datapath[i]))

                train_x.extend([np.array(Image.open(x).convert('RGB').resize((50, 50))) for x in train_temp])
                train_y.extend([i] * len(train_temp))

                test_temp = glob.glob(
                    'datasets/openloris/' + self.ns_type + '/test/task{}/{}/*.jpg'.format(batch_num, datapath[i]))

                test_x.extend([np.array(Image.open(x).convert('RGB').resize((50, 50))) for x in test_temp])
                test_y.extend([i] * len(test_temp))

            print("  --> batch{}'-dataset consisting of {} samples".format(batch_num, len(train_x)))
            print("  --> test'-dataset consisting of {} samples".format(len(test_x)))
            self.train_set.append((np.array(train_x), np.array(train_y)))
            self.test_set.append((np.array(test_x), np.array(test_y)))
        e = time.time()
        print('loading time: {}'.format(str(e - s)))

    def new_run(self, **kwargs):
        pass

    def new_task(self, cur_task, **kwargs):
        train_x, train_y = self.train_set[cur_task]
        # get val set
        train_x_rdm, train_y_rdm = shuffle_data(train_x, train_y)
        val_size = int(len(train_x_rdm) * self.params.val_size)
        val_data_rdm, val_label_rdm = train_x_rdm[:val_size], train_y_rdm[:val_size]
        train_data_rdm, train_label_rdm = train_x_rdm[val_size:], train_y_rdm[val_size:]
        self.val_set.append((val_data_rdm, val_label_rdm))
        labels = set(train_label_rdm)
        return train_data_rdm, train_label_rdm, labels

    def setup(self, **kwargs):
        pass



openloris_ntask = {
    'illumination': 9,
    'occlusion': 9,
    'pixel': 9,
    'clutter': 9,
    'sequence': 12
}

datapath = ['bottle_01', 'bottle_02', 'bottle_03', 'bottle_04', 'bowl_01', 'bowl_02', 'bowl_03', 'bowl_04', 'bowl_05',
            'corkscrew_01', 'cottonswab_01', 'cottonswab_02', 'cup_01', 'cup_02', 'cup_03', 'cup_04', 'cup_05',
            'cup_06', 'cup_07', 'cup_08', 'cup_10', 'cushion_01', 'cushion_02', 'cushion_03', 'glasses_01',
            'glasses_02', 'glasses_03', 'glasses_04', 'knife_01', 'ladle_01', 'ladle_02', 'ladle_03', 'ladle_04',
            'mask_01', 'mask_02', 'mask_03', 'mask_04', 'mask_05', 'paper_cutter_01', 'paper_cutter_02',
            'paper_cutter_03', 'paper_cutter_04', 'pencil_01', 'pencil_02', 'pencil_03', 'pencil_04', 'pencil_05',
            'plasticbag_01', 'plasticbag_02', 'plasticbag_03', 'plug_01', 'plug_02', 'plug_03', 'plug_04', 'pot_01',
            'scissors_01', 'scissors_02', 'scissors_03', 'stapler_01', 'stapler_02', 'stapler_03', 'thermometer_01',
            'thermometer_02', 'thermometer_03', 'toy_01', 'toy_02', 'toy_03', 'toy_04', 'toy_05','nail_clippers_01','nail_clippers_02',
            'nail_clippers_03', 'bracelet_01', 'bracelet_02','bracelet_03', 'comb_01','comb_02',
            'comb_03', 'umbrella_01','umbrella_02','umbrella_03','socks_01','socks_02','socks_03',
            'toothpaste_01','toothpaste_02','toothpaste_03','wallet_01','wallet_02','wallet_03',
            'headphone_01','headphone_02','headphone_03', 'key_01','key_02','key_03',
             'battery_01', 'battery_02', 'mouse_01', 'pencilcase_01', 'pencilcase_02', 'tape_01',
             'chopsticks_01', 'chopsticks_02', 'chopsticks_03',
               'notebook_01', 'notebook_02', 'notebook_03',
               'spoon_01', 'spoon_02', 'spoon_03',
               'tissue_01', 'tissue_02', 'tissue_03',
              'clamp_01', 'clamp_02', 'hat_01', 'hat_02', 'u_disk_01', 'u_disk_02', 'swimming_glasses_01'
            ]



