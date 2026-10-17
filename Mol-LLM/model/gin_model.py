import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax, to_dense_batch
from torch_geometric.nn import (
    global_add_pool,
    global_mean_pool,
    global_max_pool,
    GlobalAttention,
    Set2Set,
)
import torch.nn.functional as F

# from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros
from dataclasses import dataclass

num_atom_type = 120  # including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6  # including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3


class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not.


    See https://arxiv.org/abs/1810.00826
    """

    def __init__(self, emb_dim, aggr="add"):
        super(GINConv, self).__init__(aggr="add")
        # multi-layer perceptron
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 2 * emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * emb_dim, emb_dim),
        )
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        # print('--------------------')
        # print('x:', x.shape)
        # print('edge_index:',edge_index.shape)
        edge_index, edge_attr = add_self_loops(
            edge_index, edge_attr, fill_value=0, num_nodes=x.size(0)
        )

        # add features corresponding to self-loop edges.
        # self_loop_attr = torch.zeros(x.size(0), 2)
        # self_loop_attr[:,0] = 4 #bond type for self-loop edge
        # self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        # print('edge_attr:',edge_attr.shape)
        # print('self_loop_attr:',self_loop_attr.shape)
        # print('--------------------')
        # edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(
            edge_attr[:, 1]
        )

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GCNConv(MessagePassing):

    def __init__(self, emb_dim, aggr="add"):
        super(GCNConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def norm(self, edge_index, num_nodes, dtype):
        ## assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones(
            (edge_index.size(1),), dtype=dtype, device=edge_index.device
        )
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(
            edge_attr[:, 1]
        )

        norm = self.norm(edge_index, x.size(0), x.dtype)

        x = self.linear(x)

        return self.propagate(
            self.aggr, edge_index, x=x, edge_attr=edge_embeddings, norm=norm
        )

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)


class GATConv(MessagePassing):
    def __init__(self, emb_dim, heads=2, negative_slope=0.2, aggr="add"):
        super(GATConv, self).__init__()

        self.aggr = aggr

        self.emb_dim = emb_dim
        self.heads = heads
        self.negative_slope = negative_slope

        self.weight_linear = torch.nn.Linear(emb_dim, heads * emb_dim)
        self.att = torch.nn.Parameter(torch.Tensor(1, heads, 2 * emb_dim))

        self.bias = torch.nn.Parameter(torch.Tensor(emb_dim))

        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, heads * emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, heads * emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):

        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(
            edge_attr[:, 1]
        )

        x = self.weight_linear(x).view(-1, self.heads, self.emb_dim)
        return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, edge_index, x_i, x_j, edge_attr):
        edge_attr = edge_attr.view(-1, self.heads, self.emb_dim)
        x_j += edge_attr

        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0])

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        aggr_out = aggr_out.mean(dim=1)
        aggr_out = aggr_out + self.bias

        return aggr_out


class GraphSAGEConv(MessagePassing):
    def __init__(self, emb_dim, aggr="mean"):
        super(GraphSAGEConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(
            edge_attr[:, 1]
        )

        x = self.linear(x)

        return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return F.normalize(aggr_out, p=2, dim=-1)


class GNN(torch.nn.Module):
    """


    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """

    def __init__(
        self, num_layer, emb_dim, JK="last", drop_ratio=0, gnn_type="gin", args=None
    ):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.args = args

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        ##List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr="add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim))

        self.pool = global_mean_pool

        ##List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))
        self.num_features = emb_dim
        self.cat_grep = True

    # def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = (
                data.x,
                data.edge_index,
                data.edge_attr,
                data.batch,
            )
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            # h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        ## Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]
        else:
            raise ValueError("Invalid graph pooling type.")

        h_graph = self.pool(node_representation, batch)  # shape = [B, D]
        batch_node, batch_mask = to_dense_batch(
            node_representation, batch
        )  # shape = [B, n_max, D],
        batch_mask = batch_mask.bool()

        if self.cat_grep:
            batch_node = torch.cat(
                (h_graph.unsqueeze(1), batch_node), dim=1
            )  # shape = [B, n_max+1, D]
            batch_mask = torch.cat(
                [
                    torch.ones(
                        (batch_mask.shape[0], 1), dtype=torch.bool, device=batch.device
                    ),
                    batch_mask,
                ],
                dim=1,
            )
            return batch_node, batch_mask
        else:
            return batch_node, batch_mask, h_graph


def reverse_to_dense_batch(dense_x, mask):
    """
    Reverses the to_dense_batch operation to retrieve the original sparse format.

    Parameters:
        dense_x (Tensor): The dense batch tensor of shape (batch_size, max_num_nodes, num_features).
        mask (BoolTensor): The mask indicating valid (True) and padded (False) positions.

    Returns:
        x (Tensor): The original sparse node feature matrix.
        batch (Tensor): The original batch vector.
    """
    batch_size, max_num_nodes, num_features = dense_x.shape
    num_nodes = mask.sum(dim=1)  # Number of actual nodes in each graph

    # Initialize a list to hold the node features
    original_features = []
    # Initialize a list to hold the batch indices
    batch_indices = []

    for i in range(batch_size):
        # Extract the features of the actual nodes (where mask is True)
        actual_node_features = dense_x[i, mask[i]]
        original_features.append(actual_node_features)
        # Create a batch index for these nodes
        batch_indices.extend([i] * num_nodes[i].item())

    # Concatenate all node features and convert batch indices to a tensor
    x = torch.cat(original_features, dim=0)
    batch = torch.tensor(batch_indices, dtype=torch.long)

    return x, batch


class GNN_Decoder(GNN):
    def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0, gnn_type="gin"):
        super(GNN_Decoder, self).__init__(num_layer, emb_dim, JK, drop_ratio, gnn_type)
        self.projection_layer = torch.nn.Linear(768 * 8, emb_dim)
        self.node_feature1_head = torch.nn.Linear(emb_dim, num_atom_type)
        self.node_feature2_head = torch.nn.Linear(emb_dim, num_chirality_tag)

    def forward(self, *argv):
        assert len(argv) == 4
        batch_node, batch_mask, edge_index, edge_attr = (
            argv[0],
            argv[1],
            argv[2],
            argv[3],
        )
        batch_node = argv[0]
        batch_node = self.projection_layer(batch_node)
        batch_mask = batch_mask[
            :, 1:
        ]  # remove the first node featuer which is mean-pooled graph feature
        x, batch = reverse_to_dense_batch(batch_node, batch_mask)

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            # h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        ## Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]

        node_representation1_logit = self.node_feature1_head(node_representation)
        node_representation2_logit = self.node_feature2_head(node_representation)
        node_representation1_prob = F.softmax(node_representation1_logit, dim=-1)
        node_representation2_prob = F.softmax(node_representation2_logit, dim=-1)
        return GNNDecoderOutput(
            atom_type_prob=node_representation1_prob,
            chirality_tag_prob=node_representation2_prob,
        )


@dataclass
class GNNDecoderOutput:
    atom_type_prob: torch.Tensor
    chirality_tag_prob: torch.Tensor


class GNN_graphpred(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set
        gnn_type: gin, gcn, graphsage, gat

    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    """

    def __init__(
        self,
        num_layer,
        emb_dim,
        num_tasks,
        JK="last",
        drop_ratio=0,
        graph_pooling="mean",
        gnn_type="gin",
    ):
        super(GNN_graphpred, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type=gnn_type)

        # Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(
                    gate_nn=torch.nn.Linear((self.num_layer + 1) * emb_dim, 1)
                )
            else:
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        # For graph-level binary classification
        if graph_pooling[:-1] == "set2set":
            self.mult = 2
        else:
            self.mult = 1

        if self.JK == "concat":
            self.graph_pred_linear = torch.nn.Linear(
                self.mult * (self.num_layer + 1) * self.emb_dim, self.num_tasks
            )
        else:
            self.graph_pred_linear = torch.nn.Linear(
                self.mult * self.emb_dim, self.num_tasks
            )

    def from_pretrained(self, model_file):
        # self.gnn = GNN(self.num_layer, self.emb_dim, JK = self.JK, drop_ratio = self.drop_ratio)
        missing_keys, unexpected_keys = self.gnn.load_state_dict(torch.load(model_file))
        print(missing_keys)
        print(unexpected_keys)

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = (
                data.x,
                data.edge_index,
                data.edge_attr,
                data.batch,
            )
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.gnn(x, edge_index, edge_attr)

        return self.graph_pred_linear(self.pool(node_representation, batch))


import torch.nn as nn

from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims


class AtomEncoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(AtomEncoder, self).__init__()
        full_atom_feature_dims = get_atom_feature_dims()
        # pretrained moleculestm ckpt use different atom feature dims
        full_atom_feature_dims[1] = 4

        self.atom_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_atom_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:, i])

        return x_embedding


class BondEncoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(BondEncoder, self).__init__()
        full_bond_feature_dims = get_bond_feature_dims()

        self.bond_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_bond_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.view(-1, len(self.bond_embedding_list))
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:, i])

        return bond_embedding


class GINConv_MoleculeSTM(MessagePassing):
    def __init__(self, emb_dim, aggr="add"):
        """
        emb_dim (int): node embedding dimensionality
        """
        super(GINConv_MoleculeSTM, self).__init__(aggr=aggr)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 2 * emb_dim),
            torch.nn.BatchNorm1d(2 * emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * emb_dim, emb_dim),
        )
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        self.bond_encoder = BondEncoder(emb_dim=emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp(
            (1 + self.eps) * x
            + self.propagate(edge_index, x=x, edge_attr=edge_embedding)
        )
        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class GNN_MoleculeSTM(nn.Module):
    def __init__(
        self, num_layer, emb_dim, JK="last", drop_ratio=0.0, gnn_type="gin", args=None
    ):

        if num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        super(GNN_MoleculeSTM, self).__init__()
        self.drop_ratio = drop_ratio
        self.num_layer = num_layer
        self.JK = JK

        self.atom_encoder = AtomEncoder(emb_dim)

        ###List of MLPs
        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv_MoleculeSTM(emb_dim, aggr="add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim))

        self.pool = global_mean_pool

        ###List of batchnorms
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))
        self.num_features = emb_dim
        self.cat_grep = True

    # def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = (
                data.x,
                data.edge_index,
                data.edge_attr,
                data.batch,
            )
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.atom_encoder(x)

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            # h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]
        else:
            raise ValueError("not implemented.")

        h_graph = self.pool(node_representation, batch)  # shape = [B, D]
        batch_node, batch_mask = to_dense_batch(
            node_representation, batch
        )  # shape = [B, n_max, D],
        batch_mask = batch_mask.bool()

        if self.cat_grep:
            batch_node = torch.cat(
                (h_graph.unsqueeze(1), batch_node), dim=1
            )  # shape = [B, n_max+1, D]
            batch_mask = torch.cat(
                [
                    torch.ones(
                        (batch_mask.shape[0], 1), dtype=torch.bool, device=batch.device
                    ),
                    batch_mask,
                ],
                dim=1,
            )
            return batch_node, batch_mask
        else:
            return batch_node, batch_mask, h_graph


if __name__ == "__main__":
    pass
