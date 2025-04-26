from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.conv import MessagePassing
from torch_sparse import SparseTensor, matmul, mul
from torch_sparse import sum as sparsesum
from torch import nn, Tensor
import torch


class mymodel(MessagePassing):
    def __init__(self, users_cnt: int, items_cnt: int, config, K: int = 3,
                 add_self_loops: bool = False):
        """Initializes OurModel Model

        Args:
            users_cnt (int): Number of users
            items_cnt (int): Number of items
            embedding_dim (int, optional): Dimensionality of embeddings. Defaults to 8.
            K (int, optional): Number of message passing layers. Defaults to 3.
            add_self_loops (bool, optional): Whether to add self loops for message passing. Defaults to False.
        """

        super().__init__()
        self.name = 'mymodel'
        self.num_users, self.num_items = users_cnt, items_cnt
        self.embedding_dim, self.K = config['embedding_size'], K
        self.add_self_loops = add_self_loops
        self.loss_fn = nn.BCELoss()

        self.users_emb = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.embedding_dim)
        self.items_emb = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.embedding_dim)  # e_i^0

        nn.init.xavier_normal_(self.users_emb.weight)
        nn.init.xavier_normal_(self.items_emb.weight)
        self.config = config

    def forward(self, edge_index: SparseTensor):
        """Forward propagation of OurModel Model.

        Args:
            edge_index (SparseTensor): adjacency matrix

        Returns:
            tuple (Tensor): e_u^K, e_i^K
        """

        edge_index_norm = gcn_norm(edge_index, add_self_loops=self.add_self_loops)

        emb_0 = torch.cat([self.items_emb.weight, self.users_emb.weight])  # E^0
        embs = [emb_0]
        emb_k = emb_0

        for i in range(self.K):
            emb_k = self.propagate(edge_index_norm, x=emb_k)
            embs.append(emb_k)

        embs = torch.stack(embs, dim=1)
        emb_final = torch.mean(embs, dim=1)  # E^K

        users_emb_final, items_emb_final = torch.split(emb_final, [self.num_items, self.num_users])

        return users_emb_final, items_emb_final

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor):
        return matmul(adj_t, x)

    def cal_loss(self, users_embd: Tensor, items_embd: Tensor, lmbd: float, weight_decay: Tensor, neg_cnt, *tensors):

        users_emb_K = users_embd[tensors[0]]
        pos_items_emb_K = items_embd[tensors[1]]
        neg_items_emb_K = items_embd[tensors[2]]
        users_emb_0 = self.users_emb(tensors[0])
        pos_items_emb_0 = self.items_emb(tensors[1])
        neg_items_emb_0 = self.items_emb(tensors[2])

        reg_loss = lmbd * (
                users_emb_0.norm(2).pow(2) + pos_items_emb_0.norm(2).pow(2) + neg_items_emb_0.norm(2).pow(2)) / float(
            len(users_emb_K))

        pos_scores = torch.mul(users_emb_K, pos_items_emb_K)
        pos_scores = torch.sum(pos_scores, dim=1)  # predicted scores of positive samples
        loss = 0
        loss1 = 0
        loss2 = 0
        coef = 0.0
        for i in range(neg_cnt):
            neg_scores = torch.mul(users_emb_K, neg_items_emb_K[:, i, :])
            neg_scores = torch.sum(neg_scores, dim=1)  # predicted scores of negative samples

            m = nn.Sigmoid()

            loss1 += self.loss_fn(m(pos_scores), torch.ones(pos_scores.shape[0]).to(self.config['device']))
            loss2 += self.loss_fn(m(neg_scores), torch.zeros(neg_scores.shape[0]).to(self.config['device']))
            loss += (1 - coef) * loss1 + (1 + coef) * loss2

        loss /= neg_cnt
        loss += weight_decay * reg_loss

        return loss