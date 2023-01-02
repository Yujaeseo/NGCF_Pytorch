import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp

class NGCF(nn.Module):
    def __init__(self, n_user, n_item, norm_adj, args):
        super(NGCF, self).__init__()
        self.n_user = n_user
        self.n_item = n_item

        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.node_dropout = args.node_dropout[0]
        self.message_dropout = args.message_dropout

        self.norm_adj = norm_adj

        self.layers = eval(args.layer_size)
        self.n_layers = len(self.layers)
        self.decay = eval(args.regs)[0]

        self.embedding_dict, self.weight_dict = self.init_weight()

        self.L = self._convert_sp_mat_to_sp_tensor(self.norm_adj).to(args.device)
        self.L_I = self._convert_sp_mat_to_sp_tensor(self.norm_adj + sp.eye(self.norm_adj.shape[0])).to(args.device)

    def init_weight(self):
        initializer = nn.init.xavier_uniform_

        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.n_user, self.emb_dim))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.n_item, self.emb_dim)))
        })

        weight_dict = nn.ParameterDict()
        layers = [self.emb_dim] + self.layers

        for k in range(len(self.layers)):
            weight_dict.update({'W_gc_{}'.format(k): nn.Parameter(initializer(torch.empty(layers[k], layers[k + 1])))})
            weight_dict.update({'b_gc_{}'.format(k): nn.Parameter(initializer(torch.empty(1, layers[k + 1])))})
            weight_dict.update({'W_bi_{}'.format(k): nn.Parameter(initializer(torch.empty(layers[k], layers[k + 1])))})
            weight_dict.update({'b_bi_{}'.format(k): nn.Parameter(initializer(torch.empty(1, layers[k + 1])))})

        return embedding_dict, weight_dict

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float)

        i = torch.LongTensor(np.mat([coo.row, coo.col]))
        v = torch.FloatTensor(coo.data)

        res = torch.sparse.FloatTensor(i, v, coo.shape)
        return res

    def sparse_dropout(self, x, rate, noise_shape):
        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)

        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)

        return out * (1. / (1 - rate))

    def forward(self, users, pos_items, neg_items, drop_flag=True):

        if drop_flag:
            L_hat = self.sparse_dropout(self.L, self.node_dropout, self.L._nnz())
            L_I_hat = self.sparse_dropout(self.L_I, self.node_dropout, self.L_I._nnz())
        else:
            L_hat, L_I_hat = self.L, self.L_I

        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)

        all_embeddings = [ego_embeddings]

        for k in range(self.n_layers):
            L_side_embeddings = torch.sparse.mm(L_hat, ego_embeddings)
            L_I_side_embeddings = torch.sparse.mm(L_I_hat, ego_embeddings)

            sum_embeddings = torch.matmul(L_I_side_embeddings, self.weight_dict['W_gc_{}'.format(k)]) + \
                             self.weight_dict['b_gc_{}'.format(k)]

            bi_embeddings = torch.mul(L_side_embeddings, ego_embeddings)

            bi_embeddings = torch.matmul(bi_embeddings, self.weight_dict['W_bi_{}'.format(k)]) + self.weight_dict[
                'b_bi_{}'.format(k)]

            ego_embeddings = nn.LeakyReLU(negative_slope=0.2)(sum_embeddings + bi_embeddings)
            ego_embeddings = nn.Dropout(self.message_dropout[k])(ego_embeddings)

            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)

            all_embeddings += [norm_embeddings]

        all_embeddings = torch.cat(all_embeddings, 1)
        u_g_embeddings = all_embeddings[:self.n_user, :]
        i_g_embeddings = all_embeddings[self.n_user:, :]

        u_g_embeddings = u_g_embeddings[users, :]
        pos_i_g_embeddings = i_g_embeddings[pos_items, :]
        neg_i_g_embeddings = i_g_embeddings[neg_items, :]

        return u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings

    def bpr_loss(self, u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings):
        pos_yui = torch.sum(torch.mul(u_g_embeddings, pos_i_g_embeddings), axis=1)
        neg_yuj = torch.sum(torch.mul(u_g_embeddings, neg_i_g_embeddings), axis=1)

        maxi = nn.LogSigmoid()(pos_yui - neg_yuj)
        mf_loss = -1 * torch.mean(maxi)

        # regularizer = (torch.sum(u_g_embeddings**2) + torch.sum(pos_i_g_embeddings**2) + torch.sum(neg_i_g_embeddings**2))/2
        regularizer = (torch.norm(u_g_embeddings) ** 2
                       + torch.norm(pos_i_g_embeddings) ** 2
                       + torch.norm(neg_i_g_embeddings) ** 2) / 2
        emb_loss = (self.decay * regularizer) / self.batch_size
        bpr_loss = mf_loss + emb_loss

        return bpr_loss

    def rating(self, u_g_embeddings, pos_i_g_embeddings):
        return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())