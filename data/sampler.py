__all__ = ["PairwiseSampler"]

from collections import Iterable
from collections import defaultdict

import numpy as np

from data import Interaction_IC, Interaction_UI, Interaction_UC
from utils import DataIterator, randint_choice, typeassert


class PairwiseSampler(object):
    @typeassert(dataset_UI=Interaction_UI, dataset_IC=Interaction_IC, dataset_UC=Interaction_UC, batch_size=int,
                shuffle=bool, drop_last=bool)
    def __init__(self, dataset_UI, dataset_IC, dataset_UC, sim_prob_beta, batch_size, shuffle=True, drop_last=False):
        """
        Initializes a new `PairwiseSampler` instance.

        Args:
            dataset_UI: An instance of `Interaction_UI`.
            dataset_IC: An instance of `Interaction_IC`.
            dataset_UC: An instance of `Interaction_UC`.
            batch_size (int): How many samples per batch to load.
            shuffle (bool): Whether reshuffling the samples at every epoch. Defaults to `True`.
            drop_last (bool): Whether dropping the last incomplete batch. Defaults to `False`.
        """
        super(PairwiseSampler, self).__init__()

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.sim_prob_beta = sim_prob_beta
        self.user_pos_dict = dataset_UI.to_user_dict()
        self.item_pos_dict = dataset_UI.to_item_dict()
        self.item_cate_dict = dataset_IC.to_item_dict()
        self.cate_item_dict = dataset_IC.to_cate_dict()
        self.user_cate_dict = dataset_UC.to_user_dict()
        self.num_items = max(self.item_cate_dict.keys()) + 1
        self.num_cates = max(self.item_cate_dict.values()) + 1
        self.num_trainings = sum([len(item) for u, item in self.user_pos_dict.items()])

        self.user_cate_item_dict = {}
        for user, cates in self.user_cate_dict.items():
            self.user_cate_item_dict[user] = {}
            for cate in cates:
                self.user_cate_item_dict[user][cate] = np.array([], dtype=np.int32)
            for item in self.user_pos_dict[user]:
                cate = self.item_cate_dict[item]
                self.user_cate_item_dict[user][int(cate)] = np.append(self.user_cate_item_dict[user][int(cate)], item)

    def _pairwise_sampling_ui(self):
        user_arr = np.array(list(self.user_pos_dict.keys()), dtype=np.int32)
        user_idx = randint_choice(len(user_arr), size=self.num_trainings, replace=True)
        users_list = user_arr[user_idx]

        user_pos_len = defaultdict(int)
        for u in users_list:
            user_pos_len[u] += 1

        user_pos_sample = dict()
        user_neg_sample = dict()

        for user, pos_len in user_pos_len.items():
            user_pos_sample[user] = []
            cates_num_dict = defaultdict(int)

            cates = self.user_cate_dict[user]
            if len(cates) == 1:
                pos_idx = [0] * pos_len
            else:
                pos_idx = randint_choice(len(cates), size=pos_len, replace=True)

            pos_idx = pos_idx if isinstance(pos_idx, Iterable) else [pos_idx]

            for idx in pos_idx:
                cates_num_dict[cates[idx]] += 1

            for cate, leng in cates_num_dict.items():
                if len(self.user_cate_item_dict[user][cate]) == 1:
                    item_idx = [0] * leng
                else:
                    item_idx = randint_choice(len(self.user_cate_item_dict[user][cate]), size=leng, replace=True)
                item_idx = item_idx if isinstance(item_idx, Iterable) else [item_idx]
                user_pos_sample[user].extend(self.user_cate_item_dict[user][cate][item_idx])

            user_neg_sample[user] = list()

            for pos in user_pos_sample[user]:
                if np.random.random() < self.sim_prob_beta:
                    candidate_items = self.cate_item_dict[int(self.item_cate_dict[pos])]
                    candidate_items_list = list()
                    for candidate_item in candidate_items:
                        if candidate_item not in self.user_pos_dict[user]:
                            candidate_items_list.append(candidate_item)
                    if len(candidate_items_list) > 0:
                        neg_item = int(np.random.choice(candidate_items_list, size=1))
                    else:
                        neg_item = randint_choice(self.num_items, size=1, replace=True, exclusion=self.user_pos_dict[user])
                else:
                    neg_item = randint_choice(self.num_items, size=1, replace=True, exclusion=self.user_pos_dict[user])

                user_neg_sample[user].append(neg_item)

        pos_items_list = [user_pos_sample[user].pop() for user in users_list]
        neg_items_list = [user_neg_sample[user].pop() for user in users_list]

        return users_list, pos_items_list, neg_items_list

    def _pairwise_sampling_ic(self):
        item_arr = np.array(list(self.item_cate_dict.keys()), dtype=np.int32)
        item_idx = randint_choice(len(item_arr), size=self.num_trainings, replace=True)
        items_list = item_arr[item_idx]

        item_len = defaultdict(int)
        for i in items_list:
            item_len[i] += 1

        item_sample = dict()

        for item, pos_len in item_len.items():
            item_sample[item] = list()
            pos_cates = self.item_cate_dict[item]
            neg_cates = randint_choice(self.num_cates, size=pos_len, replace=True, exclusion=pos_cates)
            neg_cates = neg_cates if isinstance(neg_cates, Iterable) else [neg_cates]
            item_sample[item] = list(neg_cates)

        cates_list = [item_sample[item].pop() for item in items_list]

        return items_list, cates_list

    def _pairwise_sampling_uc(self):
        user_arr = np.array(list(self.user_pos_dict.keys()), dtype=np.int32)
        user_idx = randint_choice(len(user_arr), size=self.num_trainings, replace=True)
        users_list = user_arr[user_idx]

        user_pos_len = defaultdict(int)
        for u in users_list:
            user_pos_len[u] += 1

        user_sample = dict()

        for user, pos_len in user_pos_len.items():
            user_sample[user] = list()
            pos_cates = self.user_cate_dict[user]
            if len(pos_cates) == self.num_cates:
                neg_cates = randint_choice(self.num_cates, size=pos_len, replace=True)
            else:
                neg_cates = randint_choice(self.num_cates, size=pos_len, replace=True, exclusion=pos_cates)
            neg_cates = neg_cates if isinstance(neg_cates, Iterable) else [neg_cates]
            user_sample[user] = list(neg_cates)

        cates_list = [user_sample[user].pop() for user in users_list]

        return users_list, cates_list

    def __iter__(self):
        users_list, pos_items_list, neg_items_list = self._pairwise_sampling_ui()
        users_list1, cates_list1 = self._pairwise_sampling_uc()
        items_list2, cates_list2 = self._pairwise_sampling_ic()
        data_iter = DataIterator(users_list, pos_items_list, neg_items_list,
                                 users_list1, cates_list1,
                                 items_list2, cates_list2,
                                 batch_size=self.batch_size, shuffle=self.shuffle, drop_last=self.drop_last)

        for bat_users, bat_pos_items, bat_neg_items, \
        bat_users1, bat_cates1, \
        bat_items2, bat_cates2 in data_iter:
            yield np.asarray(bat_users), np.asarray(bat_pos_items), np.asarray(bat_neg_items), \
                  np.asarray(bat_users1), np.asarray(bat_cates1), \
                  np.asarray(bat_items2), np.asarray(bat_cates2)

    def __len__(self):
        n_sample = self.num_trainings
        if self.drop_last:
            return n_sample // self.batch_size
        else:
            return (n_sample + self.batch_size - 1) // self.batch_size
