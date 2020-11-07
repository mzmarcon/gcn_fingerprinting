import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, SAGEConv, BatchNorm
from utils import ContrastiveLoss
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch
from layers import TemporalGC
import numpy as np

class Siamese_GeoChebyConv(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(Siamese_GeoChebyConv,self).__init__()

        K = 3
        nclass = int(nclass)
        self.gc1 = ChebConv(nfeat, nhid, K)
        self.gc2 = ChebConv(nhid, nhid, K)
        self.gc3 = ChebConv(nhid, nhid, K)
        self.gc4 = ChebConv(nhid, nclass, K)
        self.dropout = dropout

        self.classifier = nn.Sequential(
            nn.Linear(268, 100),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(100, 60),
            # nn.ReLU(),
            # nn.Dropout(),
            # nn.Linear(60, 1),
            # nn.Sigmoid()
        )

    def forward_single(self, data):
        x = self.gc1(data['x'], edge_index=data['edge_index'], edge_weight=data['edge_attr'])
        BatchNorm(16)
        X = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc4(x, edge_index=data['edge_index'], edge_weight=data['edge_attr'])
        return x

    def forward(self, data1, data2):
        data1_graphs = data1.to_data_list()
        data2_graphs = data2.to_data_list()

        outs1 = []
        outs2 = []
        for graph_num in range(data1.num_graphs):
            input1 = data1_graphs[graph_num]
            input2 = data2_graphs[graph_num]
            
            conv_out1 = self.forward_single(input1)
            conv_out2 = self.forward_single(input2)

            dense_out1 = self.classifier(conv_out1.T)
            dense_out2 = self.classifier(conv_out2.T)

            outs1.append(dense_out1)
            outs2.append(dense_out2)

        output1 = torch.stack(outs1,dim=0)
        output2 = torch.stack(outs2,dim=0)

        return output1, output2


class Siamese_GeoCheby_Cos(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(Siamese_GeoCheby_Cos,self).__init__()

        K = 2
        nclass = int(nclass)
        self.gc1 = ChebConv(nfeat, nhid, K)
        # self.gc2 = ChebConv(2*nhid, 2*nhid, K)
        # self.gc3 = ChebConv(2*nhid, nhid, K)
        self.gc4 = ChebConv(nhid, nhid, K)
        self.dropout = dropout

        self.classifier = nn.Sequential(
            nn.Linear(4288, 1024),
            nn.ReLU(),
            nn.Dropout(),
            # nn.Linear(2048, 1024),
            # nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(1024, 512)
        )

    def forward_single(self, data):
        x = self.gc1(data['x'], edge_index=data['edge_index'], edge_weight=data['edge_attr'])
        BatchNorm(16)
        X = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        # x = self.gc2(x, edge_index=data['edge_index'], edge_weight=data['edge_attr'])
        # BatchNorm(32)
        # X = F.relu(x)
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = self.gc3(x, edge_index=data['edge_index'], edge_weight=data['edge_attr'])
        # BatchNorm(16)
        # X = F.relu(x)
        # x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc4(x, edge_index=data['edge_index'], edge_weight=data['edge_attr'])
        return x

    def forward(self, data1, data2, data3):
        data1_graphs = data1.to_data_list()
        data2_graphs = data2.to_data_list()
        data3_graphs = data3.to_data_list()

        outs1 = []
        outs2 = []
        outs3 = []
        for graph_num in range(data1.num_graphs):
            input1 = data1_graphs[graph_num]
            input2 = data2_graphs[graph_num]
            input3 = data3_graphs[graph_num]
            
            conv_out1 = self.forward_single(input1)
            conv_out2 = self.forward_single(input2)
            conv_out3 = self.forward_single(input3)

            dense_out1 = self.classifier(conv_out1.flatten())
            dense_out2 = self.classifier(conv_out2.flatten())
            dense_out3 = self.classifier(conv_out3.flatten())

            outs1.append(dense_out1)
            outs2.append(dense_out2)
            outs3.append(dense_out3)

        output1 = torch.stack(outs1,dim=0)
        output2 = torch.stack(outs2,dim=0)
        output3 = torch.stack(outs3,dim=0)

        return output1, output2, output3

class Siamese_GeoChebyConv_Read(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(Siamese_GeoChebyConv_Read,self).__init__()

        K = 3
        nclass = int(nclass)
        self.gc1 = ChebConv(nfeat, nhid, K)
        # self.gc2 = ChebConv(nhid, nhid, K)
        # self.gc3 = ChebConv(nhid, nhid, K)
        self.gc4 = ChebConv(nhid, 1, K)
        self.dropout = dropout

        self.classifier = nn.Sequential(
            nn.Linear(268, 60),
            nn.ReLU(),
            nn.Dropout(),
            # nn.Linear(100, 60),
            # nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(60, 1),
            # nn.Sigmoid()
        )

    def forward_single(self, data):
        x = self.gc1(data['x'], edge_index=data['edge_index'], edge_weight=data['edge_attr'])
        BatchNorm(16)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        # x = F.relu(self.gc2(x, edge_index=data['edge_index'], edge_weight=data['edge_attr']))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = F.relu(self.gc3(x, edge_index=data['edge_index'], edge_weight=data['edge_attr']))
        # x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc4(x, edge_index=data['edge_index'], edge_weight=data['edge_attr'])
        return x

    def forward(self, data1, data2):
        outs = []
        data1_graphs = data1.to_data_list()
        data2_graphs = data2.to_data_list()

        for graph_num in range(data1.num_graphs):
            input1 = data1_graphs[graph_num]
            input2 = data2_graphs[graph_num]
            
            conv_out1 = self.forward_single(input1)
            conv_out2 = self.forward_single(input2)

            l1_distance =  nn.PairwiseDistance(p=1.0)
            distance_out = l1_distance(conv_out1.flatten().unsqueeze(1), conv_out2.flatten().unsqueeze(1))
            dense_out = self.classifier(distance_out.T)
            outs.append(dense_out)

        output = torch.stack(outs,dim=0)

        return output
        


class Siamese_GeoSAGEConv(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(Siamese_GeoSAGEConv,self).__init__()

        nclass = int(nclass)
        self.gc1 = SAGEConv(nfeat, nhid, normalize=False)
        self.gc2 = SAGEConv(nhid, nclass, normalize=False)
        self.dropout = dropout

        self.classifier = nn.Sequential(
            nn.Linear(200, 100),
            # nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(50, 10),
            # nn.BatchNorm1d(),
            # nn.ReLU()
            # nn.Sigmoid()
        )

    def forward_single(self, data):
        # data = Data(x=features, edge_index=adj._indices())
        x = F.relu(self.gc1(data['x'], edge_index=data['edge_index'], edge_weight=data['edge_attr']))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index=data['edge_index'], edge_weight=data['edge_attr'])
        # return F.log_softmax(x, dim=1)
        return x

    def forward(self, data1, data2):
        out1 = self.forward_single(data1)
        out2 = self.forward_single(data2)
        out1 = self.classifier(out1.T)
        out2 = self.classifier(out2.T)
        # return F.log_softmax(x, dim=1)
        # return F.softmax(out1,dim=1), F.softmax(out2,dim=1)
        return out1, out2



class GeoSAGEConv(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GeoSAGEConv,self).__init__()

        nclass = int(nclass)
        self.gc1 = SAGEConv(nfeat, nhid, normalize=False)
        self.gc2 = SAGEConv(nhid, nclass, normalize=False)
        self.dropout = dropout

        self.classifier = nn.Sequential(
            nn.Linear(200, 100),
            # nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(50, 10),
            # nn.BatchNorm1d(),
            # nn.Sigmoid()
        )

    def forward(self, data):
        x = F.relu(self.gc1(data['x'], edge_index=data['edge_index'], edge_weight=data['edge_attr']))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index=data['edge_index'], edge_weight=data['edge_attr'])
        x = self.classifier(x.T)
        # return F.log_softmax(x, dim=1)
        return x


class GeoChebyConv(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GeoChebyConv,self).__init__()

        K = 3
        nclass = int(nclass)
        self.gc1 = ChebConv(nfeat, nhid, K)
        self.gc2 = ChebConv(nhid, nhid, K)
        self.gc3 = ChebConv(nhid, nhid, K)
        self.gc4 = ChebConv(nhid, nclass, K)
        self.dropout = dropout

        self.classifier = nn.Sequential(
            nn.Linear(268, 50),
            nn.ReLU(),
            nn.Dropout(),
            # nn.Linear(100, 1),
            # nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(50, 1),
        )

    def forward_single(self, data):
        x = self.gc1(data['x'], edge_index=data['edge_index'], edge_weight=data['edge_attr'])
        BatchNorm(16)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc4(x, edge_index=data['edge_index'], edge_weight=data['edge_attr'])
        return x

    def forward(self, data):
        outs = []
        data_graphs = data.to_data_list()

        for graph_num in range(data.num_graphs):
            input_graph = data_graphs[graph_num]     
            conv_out = self.forward_single(input_graph)
            dense_out = self.classifier(conv_out.T)
            outs.append(dense_out)

        output = torch.stack(outs,dim=0).squeeze()

        return output

class TemporalModel(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, in_channels, num_class, graph_args,
                 edge_importance_weighting, adj_matrix, **kwargs):
        super().__init__()

        # load graph

        A = adj_matrix
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        #Degree matrix
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i] ** (-0.5)
        #Normalized adjacency matrix
        DAD = np.dot(np.dot(Dn, A), Dn)

        temp_matrix = np.zeros((1, A.shape[0], A.shape[0]))
        temp_matrix[0] = DAD
        A = torch.tensor(temp_matrix, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks (**number of layers, final output features, kernel size**)
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 11 # update temporal kernel size
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(64, 64, kernel_size, 1, residual=False, **kwargs),
            st_gcn(64, 64, kernel_size, 1, residual=False, **kwargs),
            st_gcn(64, 64, kernel_size, 1, residual=False, **kwargs),
            #st_gcn(64, 128, kernel_size, 2, **kwargs),
            #st_gcn(128, 128, kernel_size, 1, **kwargs),
            #st_gcn(128, 128, kernel_size, 1, **kwargs),
            #st_gcn(128, 256, kernel_size, 2, **kwargs),
            #st_gcn(256, 256, kernel_size, 1, **kwargs),
            #st_gcn(256, 256, kernel_size, 1, **kwargs),
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            # self.edge_importance = nn.ParameterList([
            #     nn.Parameter(torch.ones(self.A.size()))
            #     for i in self.st_gcn_networks
            # ])
            self.edge_importance = nn.Parameter(torch.ones(self.A.size()))
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # fcn for prediction (**number of fully connected layers**)
        self.fcn = nn.Conv2d(64, num_class, kernel_size=1)
        self.sig = nn.Sigmoid()

    def forward(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        # for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
        #     x, _ = gcn(x, self.A * (importance + torch.transpose(importance,1,2)))
        #print(self.edge_importance.shape)
        for gcn in self.st_gcn_networks:
            x, _ = gcn(x, self.A * (self.edge_importance*self.edge_importance+torch.transpose(self.edge_importance*self.edge_importance,1,2)))

        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        # prediction
        # pdb.set_trace()
        x = self.fcn(x)
        x = self.sig(x)

        x = x.view(x.size(0), -1)

        return x

    def extract_feature(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        # prediction
        x = self.fcn(x)
        output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)
        # pdb.set_trace()
        return output, feature

class st_gcn(nn.Module):
    """Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0.5,
                 residual=True):
        super().__init__()
        print("Dropout={}".format(dropout))
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = TemporalGC(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A