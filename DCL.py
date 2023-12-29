__all__ = ["DCL"]

import os
import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as torch_sp

from time import time

from data import Dataset, PairwiseSampler
from utils import Logger, Configurator, Evaluator, \
                  typeassert, ensureDir, get_initializer, inner_product, l2_loss, sp_mat_to_sp_tensor


class _LightGCN(nn.Module):
    def __init__(self, num_users, num_items, num_cates, embed_dim, norm_adj_ui, norm_adj_uc, norm_adj_ic, n_layers):
        super(_LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_cates = num_cates
        self.embed_dim = embed_dim
        self.norm_adj_ui = norm_adj_ui
        self.norm_adj_uc = norm_adj_uc
        self.norm_adj_ic = norm_adj_ic
        self.n_layers = n_layers

        self.user_embeddings = nn.Embedding(self.num_users, self.embed_dim)
        self.item_embeddings = nn.Embedding(self.num_items, self.embed_dim)
        self.cate_embeddings = nn.Embedding(self.num_cates, self.embed_dim)

        self._user_embeddings_final = None
        self._item_embeddings_final = None

    def reset_parameters(self, pretrain=0, init_method="uniform", dir=None):
        if pretrain:
            pretrain_user_embedding = np.load(dir + 'user_embeddings.npy')
            pretrain_item_embedding = np.load(dir + 'item_embeddings.npy')
            pretrain_cate_embedding = np.load(dir + 'cate_embeddings.npy')

            pretrain_user_tensor = torch.FloatTensor(pretrain_user_embedding).cuda()
            pretrain_item_tensor = torch.FloatTensor(pretrain_item_embedding).cuda()
            pretrain_cate_tensor = torch.FloatTensor(pretrain_cate_embedding).cuda()

            self.user_embeddings = nn.Embedding.from_pretrained(pretrain_user_tensor)
            self.item_embeddings = nn.Embedding.from_pretrained(pretrain_item_tensor)
            self.cate_embeddings = nn.Embedding.from_pretrained(pretrain_cate_tensor)
        else:
            init = get_initializer(init_method)
            init(self.user_embeddings.weight)
            init(self.item_embeddings.weight)
            init(self.cate_embeddings.weight)

    def forward(self, users, pos_items, neg_items, users1, cates1, items2, cates2):
        user_embeddings1, item_embeddings1, user_embeddings2, cate_embeddings1, item_embeddings2, cate_embeddings2, = \
            self._forward_gcn(self.norm_adj_ui, self.norm_adj_uc, self.norm_adj_ic)

        user_embs = F.embedding(users, user_embeddings1)
        pos_item_embs = F.embedding(pos_items, item_embeddings1)
        neg_item_embs = F.embedding(neg_items, item_embeddings1)

        sup_pos_ratings = inner_product(user_embs, pos_item_embs)
        sup_neg_ratings = inner_product(user_embs, neg_item_embs)
        sup_logits = sup_pos_ratings - sup_neg_ratings

        user_embs_norm = F.embedding(users1, F.normalize(user_embeddings2, dim=1))
        cate_embs_norm1 = F.embedding(cates1, F.normalize(cate_embeddings1, dim=1))

        pos_ratings_users = inner_product(user_embs_norm, cate_embs_norm1)
        tot_ratings_users = torch.matmul(user_embs_norm, torch.transpose(cate_embs_norm1, 0, 1))
        con_logits_users = tot_ratings_users - pos_ratings_users[:, None]

        item_embs_norm = F.embedding(items2, F.normalize(item_embeddings2, dim=1))
        cate_embs_norm2 = F.embedding(cates2, F.normalize(cate_embeddings2, dim=1))

        pos_ratings_items = inner_product(item_embs_norm, cate_embs_norm2)
        tot_ratings_items = torch.matmul(item_embs_norm, torch.transpose(cate_embs_norm2, 0, 1))
        con_logits_items = tot_ratings_items - pos_ratings_items[:, None]

        return sup_logits, con_logits_users, con_logits_items

    def _forward_gcn(self, norm_adj_ui, norm_adj_uc, norm_adj_ic):
        ego_embeddings1 = torch.cat([self.user_embeddings.weight, self.item_embeddings.weight], dim=0)
        all_embeddings1 = [ego_embeddings1]
        for k in range(self.n_layers):
            ego_embeddings1 = torch_sp.mm(norm_adj_ui, ego_embeddings1)
            all_embeddings1 += [ego_embeddings1]
        all_embeddings1 = torch.stack(all_embeddings1, dim=1).mean(dim=1)
        user_embeddings1, item_embeddings1 = torch.split(all_embeddings1, [self.num_users, self.num_items], dim=0)

        ego_embeddings2 = torch.cat([self.user_embeddings.weight, self.cate_embeddings.weight], dim=0)
        all_embeddings2 = [ego_embeddings2]
        for k in range(self.n_layers):
            ego_embeddings2 = torch_sp.mm(norm_adj_uc, ego_embeddings2)
            all_embeddings2 += [ego_embeddings2]
        all_embeddings2 = torch.stack(all_embeddings2, dim=1).mean(dim=1)
        user_embeddings2, cate_embeddings1 = torch.split(all_embeddings2, [self.num_users, self.num_cates], dim=0)

        ego_embeddings3 = torch.cat([self.item_embeddings.weight, self.cate_embeddings.weight], dim=0)
        all_embeddings3 = [ego_embeddings3]
        for k in range(self.n_layers):
            ego_embeddings3 = torch_sp.mm(norm_adj_ic, ego_embeddings3)
            all_embeddings3 += [ego_embeddings3]
        all_embeddings3 = torch.stack(all_embeddings3, dim=1).mean(dim=1)
        item_embeddings2, cate_embeddings2 = torch.split(all_embeddings3, [self.num_items, self.num_cates], dim=0)

        return user_embeddings1, item_embeddings1, user_embeddings2, cate_embeddings1, item_embeddings2, cate_embeddings2

    def predict(self, users):
        if self._user_embeddings_final is None or self._item_embeddings_final is None:
            raise ValueError("Please first switch to 'eval' mode.")
        user_embs = F.embedding(users, self._user_embeddings_final)
        temp_item_embs = self._item_embeddings_final

        ratings = torch.matmul(user_embs, temp_item_embs.T)

        return ratings

    def eval(self):
        super(_LightGCN, self).eval()
        _user_embeddings_final1, _item_embeddings_final1, \
        _user_embeddings_final2, _cate_embeddings_final1, \
        _item_embeddings_final2, _cate_embeddings_final2, = \
            self._forward_gcn(self.norm_adj_ui, self.norm_adj_uc, self.norm_adj_ic)

        self._user_embeddings_final = _user_embeddings_final1 + _user_embeddings_final2
        self._item_embeddings_final = _item_embeddings_final1 + _item_embeddings_final2


class DCL(object):
    def __init__(self, config):
        super(DCL, self).__init__()
        self.dataset = Dataset(config.data_dir, config.dataset)
        self.logger = self._create_logger(config, self.dataset)

        user_train_dict = self.dataset.train_data.to_user_dict()
        user_test_dict = self.dataset.test_data.to_user_dict()
        item_cate_dict = self.dataset.item_cate_data.to_item_dict()

        self.evaluator = Evaluator(self.dataset, user_train_dict, user_test_dict, item_cate_dict, metric=config.metric,
                                   top_k=config.top_k, batch_size=config.test_batch_size, num_thread=config.test_thread)

        self.dataset_name = config["dataset"]

        # General hyper-parameters
        self.reg = config['reg']
        self.emb_size = config['embed_size']
        self.batch_size = config['batch_size']
        self.test_batch_size = config['test_batch_size']
        self.epochs = config["epochs"]
        self.verbose = config["verbose"]
        self.stop_cnt = config["stop_cnt"]
        self.learner = config["learner"]
        self.lr = config['lr']
        self.param_init = config["param_init"]
        self.sim_prob_beta = config["beta"]
        self.start_testing_epoch = config["start_testing_epoch"]

        # Hyper-parameters for GCN
        self.n_layers = config['n_layers']

        # Hyper-parameters for Contrastive Learning
        self.con_reg1 = config["con_reg1"]
        self.con_reg2 = config["con_reg2"]
        self.con_temp = config["con_temp"]

        # Other hyper-parameters
        self.best_epoch = 0
        self.best_result = np.zeros([2], dtype=float)

        self.model_str = 'layers=%d-reg=%.0e' % (self.n_layers, self.reg)
        self.model_str += '-temp=%.2f-user_cate_reg=%.0e-item_cate_reg=%.0e' % \
                          (self.con_temp, self.con_reg1, self.con_reg2)

        self.pretrain_flag = config["pretrain_flag"]
        self.save_flag = config["save_flag"]
        self.save_dir, self.tmp_model_dir = None, None
        if self.pretrain_flag or self.save_flag:
            self.tmp_model_dir = config.data_dir + '%s/model_tmp/%s/%s/' % \
                                 (self.dataset_name, self.model_name, self.model_str)
            self.save_dir = config.data_dir + '%s/pretrain-embeddings/%s/n_layers=%d/' % \
                            (self.dataset_name, self.model_name, self.n_layers)
            ensureDir(self.tmp_model_dir)
            ensureDir(self.save_dir)

        self.num_users, self.num_items, self.num_ratings, self.num_cates = \
            self.dataset.num_users, self.dataset.num_items, self.dataset.num_train_ratings, self.dataset.num_cates

        self.test_users = len(set(list(self.dataset.test_data.to_user_dict())))

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        adj_matrix_ui = sp_mat_to_sp_tensor(self.create_adj_mat_ui()).to(self.device)
        adj_matrix_uc = sp_mat_to_sp_tensor(self.create_adj_mat_uc()).to(self.device)
        adj_matrix_ic = sp_mat_to_sp_tensor(self.create_adj_mat_ic()).to(self.device)

        self.lightgcn = _LightGCN(self.num_users, self.num_items, self.num_cates, self.emb_size,
                                  adj_matrix_ui, adj_matrix_uc, adj_matrix_ic, self.n_layers).to(self.device)
        if self.pretrain_flag:
            self.lightgcn.reset_parameters(pretrain=self.pretrain_flag, dir=self.save_dir)
        else:
            self.lightgcn.reset_parameters(init_method=self.param_init)
        self.optimizer = torch.optim.Adam(self.lightgcn.parameters(), lr=self.lr)

    @typeassert(config=Configurator, dataset=Dataset)
    def _create_logger(self, config, dataset):
        timestamp = time()
        model_name = self.__class__.__name__
        data_name = dataset.data_name
        param_str = f"{config.summarize()}"

        run_id = f"{param_str[:150]}_{timestamp:.8f}"

        log_dir = os.path.join(config.root_dir + "log", data_name, model_name)
        logger_name = os.path.join(log_dir, run_id + ".log")
        logger = Logger(logger_name)

        logger.info(f"my pid: {os.getpid()}")
        logger.info(f"model: {self.__class__.__module__}")
        logger.info(self.dataset)
        logger.info(config)

        return logger

    def create_adj_mat_ui(self):
        n_nodes = self.num_users + self.num_items
        users_items = self.dataset.train_data.to_user_item_pairs()
        users_np, items_np = users_items[:, 0], users_items[:, 1]

        ratings = np.ones_like(users_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (users_np, items_np+self.num_users)), shape=(n_nodes, n_nodes))

        adj_mat = tmp_adj + tmp_adj.T

        # normalize adjacency matrix
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)

        return adj_matrix

    def create_adj_mat_uc(self):
        n_nodes = self.num_users + self.num_cates
        users_cates = self.dataset.user_cate_data.to_user_cate_pairs()
        users_np, cates_np = users_cates[:, 0], users_cates[:, 1]

        ratings = np.ones_like(users_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (users_np, cates_np+self.num_users)), shape=(n_nodes, n_nodes))

        adj_mat = tmp_adj + tmp_adj.T

        # normalize adjacency matrix
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)

        return adj_matrix

    def create_adj_mat_ic(self):
        n_nodes = self.num_cates + self.num_items
        items_cates = self.dataset.item_cate_data.to_item_cate_pairs()
        items_np, cates_np = items_cates[:, 0], items_cates[:, 1]

        ratings = np.ones_like(items_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (items_np, cates_np+self.num_items)), shape=(n_nodes, n_nodes))

        adj_mat = tmp_adj + tmp_adj.T

        # normalize adjacency matrix
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)

        return adj_matrix

    def train_model(self):
        data_iter = PairwiseSampler(self.dataset.train_data, self.dataset.item_cate_data, self.dataset.user_cate_data,
                                    self.sim_prob_beta, self.batch_size, shuffle=True)
        self.logger.info(self.evaluator.metrics_info())
        stopping_step = 0
        for epoch in range(1, self.epochs + 1):
            total_loss, total_bpr_loss, total_reg_loss = 0.0, 0.0, 0.0
            Infonce_loss_user, Infonce_loss_item = 0.0, 0.0
            training_start_time = time()

            for bat_users, bat_pos_items, bat_neg_items, \
            bat_users1, bat_cates1, \
            bat_items2, bat_cates2 in data_iter:
                bat_users = torch.from_numpy(bat_users).long().to(self.device)
                bat_pos_items = torch.from_numpy(bat_pos_items).long().to(self.device)
                bat_neg_items = torch.from_numpy(bat_neg_items).long().to(self.device)
                bat_users1 = torch.from_numpy(bat_users1).long().to(self.device)
                bat_cates1 = torch.from_numpy(bat_cates1).long().to(self.device)
                bat_items2 = torch.from_numpy(bat_items2).long().to(self.device)
                bat_cates2 = torch.from_numpy(bat_cates2).long().to(self.device)

                sup_logits, con_logits_users, con_logits_items = \
                    self.lightgcn.forward(bat_users, bat_pos_items, bat_neg_items,
                                          bat_users1, bat_cates1,
                                          bat_items2, bat_cates2)

                # BPR Loss
                bpr_loss = -5 * torch.sum(F.logsigmoid(sup_logits))

                # Reg Loss
                reg_loss = l2_loss(
                    self.lightgcn.user_embeddings(bat_users),
                    self.lightgcn.user_embeddings(bat_users1),
                    self.lightgcn.item_embeddings(bat_pos_items),
                    self.lightgcn.item_embeddings(bat_neg_items),
                    self.lightgcn.item_embeddings(bat_items2),
                    self.lightgcn.cate_embeddings(bat_cates1),
                    self.lightgcn.cate_embeddings(bat_cates2)
                )

                # InfoNCE Loss
                infonce_loss_user = torch.sum(torch.logsumexp(con_logits_users / self.con_temp, dim=1))
                infonce_loss_item = torch.sum(torch.logsumexp(con_logits_items / self.con_temp, dim=1))

                loss = bpr_loss + self.con_reg1 * infonce_loss_user + self.con_reg2 * infonce_loss_item + self.reg * reg_loss
                total_loss += loss
                total_bpr_loss += bpr_loss

                Infonce_loss_user += self.con_reg1 * infonce_loss_user
                Infonce_loss_item += self.con_reg2 * infonce_loss_item

                total_reg_loss += self.reg * reg_loss
                self.optimizer.zero_grad()
                loss.requires_grad_(True)
                loss.backward()
                self.optimizer.step()

            self.logger.info("[iter %d : loss : %.4f = %.4f + %.4f + %.4f + %.4f, time: %f]" % (
                epoch,
                total_loss / self.num_ratings,
                total_bpr_loss / self.num_ratings,
                Infonce_loss_user / self.num_ratings,
                Infonce_loss_item / self.num_ratings,
                total_reg_loss / self.num_ratings,
                time() - training_start_time,))

            if epoch % self.verbose == 0 and epoch > self.start_testing_epoch:
                result, flag = self.evaluate_model()
                self.logger.info("epoch %d:\t%s" % (epoch, result))
                if flag:
                    self.best_epoch = epoch
                    stopping_step = 0
                    self.logger.info("Find a better model.")
                    if self.save_flag:
                        self.logger.info("Save model to file as pretrain.")
                        torch.save(self.lightgcn.state_dict(), self.tmp_model_dir + "epoch=%d.pth" % epoch)
                else:
                   stopping_step += 1
                   if stopping_step >= self.stop_cnt:
                       self.logger.info("Early stopping is trigger at epoch: {}".format(epoch))
                       break

        self.logger.info("best_result@epoch %d:\n" % self.best_epoch)
        if self.save_flag:
            self.logger.info('Loading from the saved best model during the training process.')
            self.lightgcn.load_state_dict(torch.load(self.tmp_model_dir + "epoch=%d.pth" % self.best_epoch))
            uebd = self.lightgcn.user_embeddings.weight.cpu().detach().numpy()
            iebd = self.lightgcn.item_embeddings.weight.cpu().detach().numpy()
            cebd = self.lightgcn.cate_embeddings.weight.cpu().detach().numpy()
            np.save(self.save_dir + 'user_embeddings.npy', uebd)
            np.save(self.save_dir + 'item_embeddings.npy', iebd)
            np.save(self.save_dir + 'cate_embeddings.npy', cebd)
            buf, _ = self.evaluate_model()
        elif self.pretrain_flag:
            buf, _ = self.evaluate_model()
        else:
            buf = '\t'.join([("%.4f" % x).ljust(12) for x in self.best_result])
        self.logger.info("\t\t%s" % buf)

    def evaluate_model(self):
        flag = False
        self.lightgcn.eval()
        current_result, buf, cov, entro, gini, hit_ratio = self.evaluator.evaluate(self)
        self.logger.info("coverage: %f, entropy: %f, gini: %f, hit_ratio: %f" %
                         (cov / self.test_users, entro / self.test_users, gini / self.test_users, hit_ratio ))
        if self.best_result[1] < current_result[1]:
            self.best_result = current_result
            flag = True
        return buf, flag

    def predict(self, users):
        users = torch.from_numpy(np.asarray(users)).long().to(self.device)
        return self.lightgcn.predict(users).cpu().detach().numpy()
