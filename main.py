import numpy as np
import torch
import torch.nn as nn
# from torch.utils.data import DataLoader
from torch_geometric.data import DataLoader
from torch.autograd import Variable
from dataset_loader import ACERTA_data
from torch import autograd
from models import Siamese_GeoChebyConv, GeoSAGEConv, Siamese_GeoSAGEConv
from utils import ContrastiveLoss
import sys
from collections import defaultdict
import torch.nn.functional as F
import matplotlib.pyplot as plt

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:',device)
    
    checkpoint = 'checkpoints/checkpoint.pth'
    
    params = { 'model': 'gcn_cheby',
               'train_batch_size': 1,
               'test_batch_size': 1,
               'learning_rate': 1e-5,
               'weight_decay': 1e-1,
               'epochs': 1000,
               'early_stop': 10,
               'dropout': 0.5}
    
    training_set = ACERTA_data(set='training', split=0.8)
    test_set = ACERTA_data(set='test', split=0.8)
    
    train_loader = DataLoader(training_set, shuffle=True, drop_last=True,
                                batch_size=params['train_batch_size'])
    test_loader = DataLoader(test_set, shuffle=False, drop_last=False,
                                batch_size=params['test_batch_size'])
    

    nfeat = train_loader.__iter__().__next__()['input1']['x'].shape[1]

    criterion = ContrastiveLoss(margin=2.0)
    
    if params['model'] == 'gcn':
        model = GCN(nfeat=nfeat,
                    nhid=params['hidden'],
                    nclass=2,
                    dropout=params['dropout'])

    if params['model'] == 'sage':
        model = Siamese_GeoSAGEConv(nfeat=nfeat,
                            nhid=32,
                            nclass=1,
                            dropout=params['dropout'])

    if params['model'] == 'gcn_cheby':
        model = Siamese_GeoChebyConv(nfeat=nfeat,
                                     nhid=32,
                                     nclass=1,
                                     dropout=params['dropout'])
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'],
                                weight_decay=params['weight_decay'])


    counter=0
    match_losses = []
    nomatch_losses = []
    delta_loss = defaultdict(list)
    accuracy_list = []
    for e in range(params['epochs']):
        print('Epoch: {}/{}'.format(e,params['epochs']))

        model.train()
        label_match = torch.FloatTensor([0])
        label_nomatch = torch.FloatTensor([1])
        for i, data in enumerate(train_loader):
            input_data1 = data['input1'].to(device)
            input_data_match = data['input_match'].to(device)
            input_data_nomatch = data['input_nomatch'].to(device)

            #Match pair:
            out1, out2 = model(input_data1,input_data_match)
            # out1 = model(input_data1)
            # out2 = model(input_data_match)
            label = label_match.to(device)

            loss = criterion(out1, out2, label)
            match_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            #No-match pair:
            out1, out2 = model(input_data1,input_data_nomatch)
            # out1 = model(input_data1)
            # out2 = model(input_data_nomatch)
            label = label_nomatch.to(device)

            loss = criterion(out1, out2, label)
            nomatch_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            counter += 1
       
        model.eval()
        correct=0
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                # if not i==2: continue
                input_data1 = data['input1'].to(device)
                label = len(data['other'])-1
                similarities = defaultdict(list)

                #Compare example to each example in the test set:
                for n in range(len(data['other'])):
                    input_test = data['other'][n].to(device)

                    #Get pair similarity:
                    out1, out2 = model(input_data1,input_test)
                    similarity = F.pairwise_distance(out1, out2)
                    similarities[n].append(similarity)
                # print('Similarities:\n',similarities)

                # if not i==2: continue
                prediction = min(similarities, key=similarities.get)
                # print("Sub :{} Pred:{} True: {}".format(i,prediction,label))
                # print("Sub :{} Pred:{} True: {}".format(i,similarities[prediction],similarities[label]))
                delta = torch.abs(similarities[prediction][0]-similarities[label][0])
                print("Diff delta: ",delta)
                delta_loss[i].append(delta)
                if prediction == label:
                    correct = correct+1
                    print("Correct")
                else:
                    print("Fail")
    
            accuracy = correct/len(test_set)
            accuracy_list.append(accuracy)

            print('Training Match Loss: {:.3f}'.format(np.mean(match_losses[-1])))
            print('Training No-Match Loss: {:.3f}'.format(np.mean(nomatch_losses[-1])))
            print('Test Accuracy: {:.3f}'.format(accuracy))
            print('----------------------')
        
    
    plt.plot(range(counter),match_losses, label='Match loss')
    plt.plot(range(counter),nomatch_losses, label='No-match loss')
    plt.title('Match vs No-match Loss') 
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    for key in [k for (k,v) in delta_loss.items()]:
        plt.plot(delta_loss[key])
        plt.title('Similarity Delta Prediction vs True Label')
        plt.xlabel('Iterations')
        plt.ylabel('Similarity difference')
    plt.show()

    plt.plot(range(e+1),accuracy_list)
    plt.title('Accuracy per epoch') 
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()

    # torch.save(model.state_dict(), checkpoint)
