import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, SAGEConv, BatchNorm
from utils import ContrastiveLoss
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch 

class Siamese_GeoChebyConv(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(Siamese_GeoChebyConv,self).__init__()

        K = 3
        nclass = int(nclass)
        self.gc1 = ChebConv(nfeat, nhid, K)
        self.gc2 = ChebConv(nhid, 2*nhid, K)
        self.gc3 = ChebConv(2*nhid, nhid, K)
        self.gc4 = ChebConv(nhid, nclass, K)
        self.dropout = dropout

        self.classifier = nn.Sequential(
            nn.Linear(268, 100),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(100, 60),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(60, 1),
            # nn.Sigmoid()
        )

    def forward_single(self, data):
        x = F.relu(self.gc1(data['x'], edge_index=data['edge_index'], edge_weight=data['edge_attr']))
        x = F.dropout(x, self.dropout, training=self.training)
        # x = F.relu(self.gc2(x, edge_index=data['edge_index'], edge_weight=data['edge_attr']))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = F.relu(self.gc3(x, edge_index=data['edge_index'], edge_weight=data['edge_attr']))
        # x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc4(x, edge_index=data['edge_index'], edge_weight=data['edge_attr']))
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


class Siamese_GlobalCheby(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(Siamese_GlobalCheby,self).__init__()

        K = 3
        nclass = int(nclass)
        self.gc1 = ChebConv(nfeat, nhid, K)
        self.gc2 = ChebConv(nhid, 2*nhid, K)
        self.gc3 = ChebConv(2*nhid, nhid, K)
        self.gc4 = ChebConv(nhid, nclass, K)
        self.dropout = dropout

        self.classifier = nn.Sequential(
            nn.Linear(268, 100),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(100, 60),
            # nn.ReLU(),
            # nn.Dropout(),
            # nn.Linear(50, 10),
            nn.Sigmoid()
        )

    def forward_single(self, data):
        x = F.relu(self.gc1(data['x'], edge_index=data['edge_index'], edge_weight=data['edge_attr']))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc4(x, edge_index=data['edge_index'], edge_weight=data['edge_attr']))
        return x

    def forward(self, data1, data2):
        out1 = self.forward_single(data1)
        out2 = self.forward_single(data2)

        out1 = self.classifier(out1.T)
        out2 = self.classifier(out2.T)

        similarity = F.pairwise_distance(out1, out2)

        return similarity


class Siamese_GeoChebyConv_Read(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(Siamese_GeoChebyConv_Read,self).__init__()

        K = 3
        nclass = int(nclass)
        self.gc1 = ChebConv(nfeat, nhid, K)
        self.gc2 = ChebConv(nhid, 2*nhid, K)
        self.gc3 = ChebConv(2*nhid, nhid, K)
        self.gc4 = ChebConv(nhid, nclass, K)
        self.dropout = dropout

        self.classifier = nn.Sequential(
            nn.Linear(268, 100),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(100, 60),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(60, 1),
            # nn.Sigmoid()
        )

    def forward_single(self, data):
        x = F.relu(self.gc1(data['x'], edge_index=data['edge_index'], edge_weight=data['edge_attr']))
        x = F.dropout(x, self.dropout, training=self.training)
        # x = F.relu(self.gc2(x, edge_index=data['edge_index'], edge_weight=data['edge_attr']))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = F.relu(self.gc3(x, edge_index=data['edge_index'], edge_weight=data['edge_attr']))
        # x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc4(x, edge_index=data['edge_index'], edge_weight=data['edge_attr']))
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
            distance_out = l1_distance(conv_out1, conv_out2)
            dense_out = self.classifier(distance_out.T)
            outs.append(dense_out)

        output = torch.stack(outs,dim=0)

        return output
        

class Siamese_HingeCheby(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(Siamese_HingeCheby,self).__init__()

        K = 3
        nclass = int(nclass)
        self.gc1 = ChebConv(nfeat, nhid, K)
        self.gc2 = ChebConv(nhid, 2*nhid, K)
        self.gc3 = ChebConv(2*nhid, nhid, K)
        self.gc4 = ChebConv(nhid, nclass, K)
        self.dropout = dropout

        self.classifier = nn.Sequential(
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(100, 1),
            # nn.ReLU(),
            # nn.Dropout(),
            # nn.Linear(50, 10),
        )

    def forward_single(self, data):
        x = F.relu(self.gc1(data['x'], edge_index=data['edge_index'], edge_weight=data['edge_attr']))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc4(x, edge_index=data['edge_index'], edge_weight=data['edge_attr']))
        return x

    def forward(self, data1, data2):
        out1 = self.forward_single(data1)
        out2 = self.forward_single(data2)
        
        out_product = torch.mul(out1,out2) 
        out = self.classifier(out_product.T)
        
        return out


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
            nn.Linear(268, 100),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(100, 60),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(60, 1),
            # nn.Sigmoid()
        )

    def forward_single(self, data):
        x = F.relu(self.gc1(data['x'], edge_index=data['edge_index'], edge_weight=data['edge_attr']))
        BatchNorm(64)
        x = F.dropout(x, self.dropout, training=self.training)
        # x = F.relu(self.gc2(x, edge_index=data['edge_index'], edge_weight=data['edge_attr']))
        # BatchNorm(64)
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = F.relu(self.gc3(x, edge_index=data['edge_index'], edge_weight=data['edge_attr']))
        # BatchNorm(64)
        # x = F.dropout(x, self.dropout, training=self.training)
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