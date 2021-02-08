import random
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import gaussian
from continuum.data_utils import train_val_test_split_ni


class Original(object):
    def __init__(self, x, y, unroll=False, color=False):
        if color:
            self.x = x / 255.0
        else:
            self.x = x
        self.next_x = self.x
        self.next_y = y
        self.y = y
        self.unroll = unroll

    def get_dims(self):
        # Get data input and output dimensions
        print("input size {}\noutput size {}".format(self.x.shape[1], self.y.shape[1]))
        return self.x.shape[1], self.y.shape[1]

    def show_sample(self, num_plot=1):
        # idx = np.random.choice(self.x.shape[0])
        for i in range(num_plot):
            plt.subplot(1, 2, 1)
            if self.x[i].shape[2] == 1:
                plt.imshow(np.squeeze(self.x[i]))
            else:
                plt.imshow(self.x[i])
            plt.title("original task image")
            plt.subplot(1, 2, 2)
            if self.x[i].shape[2] == 1:
                plt.imshow(np.squeeze(self.next_x[i]))
            else:
                plt.imshow(self.next_x[i])
            plt.title(self.get_name())
            plt.axis('off')
            plt.show()

    def create_output(self):
        if self.unroll:
            ret = self.next_x.reshape((-1, self.x.shape[1] ** 2)), self.next_y
        else:
            ret = self.next_x, self.next_y
        return ret

    @staticmethod
    def clip_minmax(l, min_=0., max_=1.):
        return np.clip(l, min_, max_)

    def get_name(self):
        if hasattr(self, 'factor'):
            return str(self.__class__.__name__) + '_' + str(self.factor)

    def next_task(self, *args):
        self.next_x = self.x
        self.next_y = self.y
        return self.create_output()


class Noisy(Original):
    def __init__(self, x, y, full=False, color=False):
        super(Noisy, self).__init__(x, y, full, color)

    def next_task(self, noise_factor=0.8, sig=0.1, noise_type='Gaussian'):
        next_x = deepcopy(self.x)
        self.factor = noise_factor
        if noise_type == 'Gaussian':
            self.next_x = next_x + noise_factor * np.random.normal(loc=0.0, scale=sig, size=next_x.shape)
        elif noise_factor == 'S&P':
            # TODO implement S&P
            pass

        self.next_x = super().clip_minmax(self.next_x, 0, 1)

        return super().create_output()


class Blurring(Original):
    def __init__(self, x, y, full=False, color=False):
        super(Blurring, self).__init__(x, y, full, color)

    def next_task(self, blurry_factor=0.6, blurry_type='Gaussian'):
        next_x = deepcopy(self.x)
        self.factor = blurry_factor
        if blurry_type == 'Gaussian':
            self.next_x = gaussian(next_x, sigma=blurry_factor, multichannel=True)
        elif blurry_type == 'Average':
            pass
            # TODO implement average

        self.next_x = super().clip_minmax(self.next_x, 0, 1)

        return super().create_output()


class Occlusion(Original):
    def __init__(self, x, y, full=False, color=False):
        super(Occlusion, self).__init__(x, y, full, color)

    def next_task(self, occlusion_factor=0.2):
        next_x = deepcopy(self.x)
        self.factor = occlusion_factor
        self.image_size = next_x.shape[1]

        occlusion_size = int(occlusion_factor * self.image_size)
        half_size = occlusion_size // 2
        occlusion_x = random.randint(min(half_size, self.image_size - half_size),
                                     max(half_size, self.image_size - half_size))
        occlusion_y = random.randint(min(half_size, self.image_size - half_size),
                                     max(half_size, self.image_size - half_size))

        # self.next_x = next_x.reshape((-1, self.image_size, self.image_size))

        next_x[:, max((occlusion_x - half_size), 0):min((occlusion_x + half_size), self.image_size), \
        max((occlusion_y - half_size), 0):min((occlusion_y + half_size), self.image_size)] = 1

        self.next_x = next_x
        super().clip_minmax(self.next_x, 0, 1)

        return super().create_output()


def test_ns(x, y, ns_type, change_factor):
    ns_match = {'noise': Noisy, 'occlusion': Occlusion, 'blur': Blurring}
    change = ns_match[ns_type]
    tmp = change(x, y, color=True)
    tmp.next_task(change_factor)
    tmp.show_sample(10)


ns_match = {'noise': Noisy, 'occlusion': Occlusion, 'blur': Blurring}


def construct_ns_single(train_x_split, train_y_split, test_x_split, test_y_split, ns_type, change_factor, ns_task,
                        plot=True):
    # Data splits
    train_list = []
    test_list = []
    change = ns_match[ns_type]
    i = 0
    if len(change_factor) == 1:
        change_factor = change_factor[0]
    for idx, val in enumerate(ns_task):
        if idx % 2 == 0:
            for _ in range(val):
                print(i, 'normal')
                # train
                tmp = Original(train_x_split[i], train_y_split[i], color=True)
                train_list.append(tmp.next_task())
                if plot:
                    tmp.show_sample()

                # test
                tmp_test = Original(test_x_split[i], test_y_split[i], color=True)
                test_list.append(tmp_test.next_task())
                if plot:
                    tmp_test.show_sample()

                i += 1
        else:
            for _ in range(val):
                print(i, 'change')
                # train
                tmp = change(train_x_split[i], train_y_split[i], color=True)
                train_list.append(tmp.next_task(change_factor))
                if plot:
                    tmp.show_sample()
                # test
                tmp_test = change(test_x_split[i], test_y_split[i], color=True)
                test_list.append(tmp_test.next_task(change_factor))
                if plot:
                    tmp_test.show_sample()

                i += 1
    return train_list, test_list


def construct_ns_multiple(train_x_split, train_y_split, val_x_rdm_split, val_y_rdm_split, test_x_split,
                          test_y_split, ns_type, change_factors, plot):
    train_list = []
    val_list = []
    test_list = []
    ns_len = len(change_factors)
    for i in range(ns_len):
        factor = change_factors[i]
        if factor == 0:
            ns_generator = Original
        else:
            ns_generator = ns_match[ns_type]
        print(i, factor)
        # train
        tmp = ns_generator(train_x_split[i], train_y_split[i], color=True)
        train_list.append(tmp.next_task(factor))
        if plot:
            tmp.show_sample()

        tmp_val = ns_generator(val_x_rdm_split[i], val_y_rdm_split[i], color=True)
        val_list.append(tmp_val.next_task(factor))

        tmp_test = ns_generator(test_x_split[i], test_y_split[i], color=True)
        test_list.append(tmp_test.next_task(factor))
    return train_list, val_list, test_list


def construct_ns_multiple_wrapper(train_data, train_label, test_data, est_label, task_nums, img_size,
                                  val_size, ns_type, ns_factor, plot):
    train_data_rdm_split, train_label_rdm_split, val_data_rdm_split, val_label_rdm_split, test_data_rdm_split, test_label_rdm_split = train_val_test_split_ni(
        train_data, train_label, test_data, est_label, task_nums, img_size,
        val_size)
    train_set, val_set, test_set = construct_ns_multiple(train_data_rdm_split, train_label_rdm_split,
                                                         val_data_rdm_split, val_label_rdm_split,
                                                         test_data_rdm_split, test_label_rdm_split,
                                                         ns_type,
                                                         ns_factor,
                                                         plot=plot)
    return train_set, val_set, test_set
