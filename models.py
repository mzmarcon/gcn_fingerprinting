import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, SAGEConv
from utils import ContrastiveLoss
from torch_geometric.data import Data

class Siamese_GeoChebyConv(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(Siamese_GeoChebyConv,self).__init__()

        K = 3
        nclass = int(nclass)
        self.gc1 = ChebConv(nfeat, nhid, K)
        # self.gc2 = ChebConv(nhid, 2*nhid, K)
        # self.gc3 = ChebConv(2*nhid, nhid, K)
        self.gc4 = ChebConv(nhid, nclass, K)
        self.dropout = dropout

        self.classifier = nn.Sequential(
            nn.Linear(200, 100),
            # nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(50, 1),
            # nn.BatchNorm1d(),
        )

    def forward_single(self, data):
        x = F.relu(self.gc1(data['x'], edge_index=data['edge_index'], edge_weight=data['edge_attr']))
        x = F.dropout(x, self.dropout, training=self.training)
        # x = self.gc2(x, edge_index=data['edge_index'], edge_weight=data['edge_attr'])
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = self.gc3(x, edge_index=data['edge_index'], edge_weight=data['edge_attr'])
        # x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc4(x, edge_index=data['edge_index'], edge_weight=data['edge_attr'])
        # return F.log_softmax(x, dim=1)
        return x

    def forward(self, data1, data2):
        out1 = self.forward_single(data1)
        out2 = self.forward_single(data2)

        out1 = self.classifier(out1.T)
        out2 = self.classifier(out2.T)
        # dis = torch.abs(out1 - out2)
        # out = nn.Linear(dis)
        # return F.log_softmax(x, dim=1)
        return out1, out2


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
