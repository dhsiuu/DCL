__all__ = ["Dataset", "Interaction_UI", "Interaction_IC", "Interaction_UC"]

import os
from collections import OrderedDict

import numpy as np
import pandas as pd

from utils import typeassert


class Interaction_UI(object):
    @typeassert(data=pd.DataFrame)
    def __init__(self, data):
        self._data = data
        self.len = len(data)

    def to_user_item_pairs(self):
        return self._data[["user", "item"]].to_numpy(copy=True, dtype=np.int32)

    def to_user_dict(self):
        user_dict = OrderedDict()
        user_grouped = self._data.groupby("user")
        for user, user_data in user_grouped:
            user_dict[user] = user_data["item"].to_numpy(dtype=np.int32)

        return user_dict

    def to_item_dict(self):
        item_dict = OrderedDict()
        item_grouped = self._data.groupby("item")
        for item, item_data in item_grouped:
            item_dict[item] = item_data["user"].to_numpy(dtype=np.int32)

        return item_dict


class Interaction_IC(object):
    @typeassert(data=pd.DataFrame)
    def __init__(self, data):
        self._data = data
        self.len = len(data)

    def to_item_cate_pairs(self):
        return self._data[["item", "cate"]].to_numpy(copy=True, dtype=np.int32)

    def to_item_dict(self):
        item_dict = OrderedDict()
        item_grouped = self._data.groupby("item")
        for item, item_data in item_grouped:
            item_dict[item] = item_data["cate"].to_numpy(dtype=np.int32)

        return item_dict

    def to_cate_dict(self):
        cate_dict = OrderedDict()
        cate_grouped = self._data.groupby("cate")
        for cate, cate_data in cate_grouped:
            cate_dict[cate] = cate_data["item"].to_numpy(dtype=np.int32)

        return cate_dict


class Interaction_UC(object):
    @typeassert(data=pd.DataFrame)
    def __init__(self, data):
        self._data = data
        self.len = len(data)

    def to_user_cate_pairs(self):
        return self._data[["user", "cate"]].to_numpy(copy=True, dtype=np.int32)

    def to_user_dict(self):
        user_dict = OrderedDict()
        user_grouped = self._data.groupby("user")
        for user, user_data in user_grouped:
            user_dict[user] = user_data["cate"].to_numpy(dtype=np.int32)

        return user_dict

    def to_cate_dict(self):
        cate_dict = OrderedDict()
        cate_grouped = self._data.groupby("cate")
        for cate, cate_data in cate_grouped:
            cate_dict[cate] = cate_data["user"].to_numpy(dtype=np.int32)

        return cate_dict


class Dataset(object):
    def __init__(self, data_dir, dataset_name):
        """
        Dataset

        Notes:
            The prefix name of data files is same as the data_dir, and the
            suffix/extension names are 'train', 'test', 'valid', 'cate'.
            Directory structure:
                data_dir
                    ├── data_dir.train      // training data
                    ├── data_dir.test       // test data
                    ├── data_dir.item       // item-category data
                    ├── data_dir.user       // user-category data
        """

        self._data_dir = os.path.join(data_dir, dataset_name)
        self.data_name = dataset_name

        file_prefix = os.path.join(self._data_dir, self.data_name)

        # load data
        _train_data = pd.read_csv(file_prefix + ".train", sep=',', header=None, names=["user", "item"])
        _test_data = pd.read_csv(file_prefix + ".test", sep=',', header=None, names=["user", "item"])
        _user_data = pd.read_csv(file_prefix + ".user", sep=',', header=None, names=["user", "cate"])
        _item_data = pd.read_csv(file_prefix + ".item", sep=',', header=None, names=["item", "cate"])

        # statistical information
        data_list = [data for data in [_train_data, _test_data] if not data.empty]
        all_data = pd.concat(data_list)

        self.num_users = max(all_data["user"]) + 1
        self.num_items = max(all_data["item"]) + 1
        self.num_cates = max(_item_data["cate"]) + 1
        self.num_ratings = len(all_data)
        self.num_train_ratings = len(_train_data)

        # convert to the object of Interaction
        self.train_data = Interaction_UI(_train_data)
        self.test_data = Interaction_UI(_test_data)
        self.item_cate_data = Interaction_IC(_item_data)
        self.user_cate_data = Interaction_UC(_user_data)

    def __str__(self):
        """
        The statistic of dataset.

        Returns:
            str: The summary of statistic
        """
        sparsity = 1 - 1.0 * self.num_ratings / (self.num_users * self.num_items)

        statistic = ["Dataset statistics:",
                     "Name: %s" % self.data_name,
                     "The number of users: %d" % self.num_users,
                     "The number of items: %d" % self.num_items,
                     "The number of ratings: %d" % self.num_ratings,
                     "The number of training: %d" % self.train_data.len,
                     "The number of testing: %d" % self.test_data.len,
                     "The number of cates: %d" % self.num_cates,
                     "Average actions of users: %.2f" % (1.0 * self.num_ratings / self.num_users),
                     "Average actions of items: %.2f" % (1.0 * self.num_ratings / self.num_items),
                     "The sparsity of the dataset: %.6f%%" % (sparsity * 100)
                     ]

        return "\n".join(statistic)

    def __repr__(self):
        return self.__str__()
