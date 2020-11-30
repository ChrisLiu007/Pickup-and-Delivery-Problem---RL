import torch
import numpy as np
from torch import nn
from torch_geometric.utils import softmax, dense_to_sparse
from torch_geometric.nn import MessagePassing
import math
from nets.graph_encoder import Normalization, SkipConnection


class mySequential(nn.Sequential):
    def forward(self, *input):
        for module in self._modules.values():
            if isinstance(module, GATLayer):
                input = module(*input)
            elif isinstance(module, SkipConnection):
                if isinstance(module.module, GATConvclass):
                    input = module(input)
                else:
                    input = module(input[0]), input[1]
            else:
                input = module(input[0]), input[1]
        return input


class GATConvclass(MessagePassing):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(GATConvclass, self).__init__(aggr='add', node_dim=0)

        if val_dim is None:
            assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        if embed_dim is not None:
            self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def message(self, edge_index_i, V_i, Q_j, K_i, size_i):
        attn = self.norm_factor * (Q_j * K_i).sum(dim=-1)

        attn = softmax(attn, edge_index_i, num_nodes=size_i)

        return V_i * attn.view(-1, self.n_heads, 1)
    def update(self, aggr_out):
        out = torch.mm(
            aggr_out.contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(self.batch_size, self.graph_size, self.embed_dim)

        return out
    def forward(self, x, edge_index):
        # h should be (batch_size, graph_size, input_dim)
        self.batch_size, self.graph_size, self.input_dim = x.size()

        hflat = x.contiguous().view(-1, self.input_dim)
        qflat = x.contiguous().view(-1, self.input_dim)

        shp = shp_q = (self.batch_size*self.graph_size, self.n_heads, -1)

        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q) # other nodes that we want the attention from.
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp) # key value for each of the nodes
        V = torch.matmul(hflat, self.W_val).view(shp) # value of each of the nodes.
        out = self.propagate(edge_index, x=V, Q=Q, K=K, V=V, size=None)
        return out

class GATLayer(mySequential):
    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden=512,
            normalization='batch',
    ):
        super(GATLayer, self).__init__(
            SkipConnection(
                GATConvclass(
                    n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim
                )
            ),
            Normalization(embed_dim, normalization),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
            ),
            Normalization(embed_dim, normalization)
        )


class TorchGeoGraphAttentionEncoder(nn.Module):
    def __init__(
            self,
            n_heads,
            embed_dim,
            n_layers,
            normalization='batch',
            feed_forward_hidden=512
    ):
        super(TorchGeoGraphAttentionEncoder, self).__init__()
        self.layers = mySequential(*(
            GATLayer(n_heads, embed_dim, feed_forward_hidden, normalization)
            for _ in range(n_layers)
        ))

    def forward(self, x, edge_index):

        h = self.layers(x, edge_index)
        return h[0]
