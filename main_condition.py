import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from torch.autograd import Variable
from dataset_loader import ACERTA_FP
from torch import autograd
from models import Siamese_GeoChebyConv, GeoSAGEConv, Siamese_GeoSAGEConv
from utils import ContrastiveLoss
import sys
from collections import defaultdict
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.stats import mode 
from tqdm import tqdm

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:',device)
    
    checkpoint = 'checkpoints/checkpoint.pth'
    
    params = { 'model': 'gcn_cheby',
               'train_batch_size': 8,
               'test_batch_size': 1,
               'learning_rate': 5e-4,
               'weight_decay': 1e-1,
               'epochs': 200,
               'early_stop': 10,
               'dropout': 0.5,
               'loss_margin': 0.5,
               'input_type': 'condition', #if 'condition', input are betas for condition. if 'allbetas', input vector with all betas.
               'condition': 'pse', #set type of input condition. 'irr', 'pse', 'reg' or 'all'.
               'adj_threshold': 0.5,
               'voting_examples': 1}


    training_set = ACERTA_FP(set='training', split=0.8, input_type=params['input_type'],
                            condition=params['condition'], adj_threshold=params['adj_threshold'])

    test_set = ACERTA_FP(set='test', split=0.8, input_type=params['input_type'],
                            condition=params['condition'], adj_threshold=params['adj_threshold'])
    
    train_loader = DataLoader(training_set, shuffle=True, drop_last=True,
                                batch_size=params['train_batch_size'])

    test_loader = DataLoader(test_set, shuffle=False, drop_last=False,
                                batch_size=params['test_batch_size'])
    

    nfeat = train_loader.__iter__().__next__()['input_anchor']['x'].shape[1]
    print("NFEAT: ",nfeat)
    
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
    
    criterion = ContrastiveLoss(params['loss_margin'])

    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'],
                                weight_decay=params['weight_decay'])

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                             milestones=[20,40,60,100,150,450,1000,1500], gamma=0.25)

#Training-----------------------------------------------------------------------------

    delta_loss = defaultdict(list)
    accuracy_list = []
    training_losses = []
    counter=0   
    mean_delta_list =[]

    for e in range(params['epochs']):
        model.train()
        epoch_loss = []
        for i, data in enumerate(tqdm(train_loader)):
            input_anchor = data['input_anchor'].to(device)
            input_pair = data['input_pair'].to(device)
            label = data['label'].to(device)
            label = torch.split(label,input_anchor.num_graphs)

            #Match pair:
            out1, out2 = model(input_anchor,input_pair)

            training_loss = criterion(out1[0], out2[0], label[0])
            epoch_loss.append(training_loss.item())
            optimizer.zero_grad()
            training_loss.backward()
            optimizer.step()

        counter += 1
        training_losses.append(epoch_loss)
        lr_scheduler.step()

#Testing-----------------------------------------------------------------------------
        model.eval()
        correct=0
        seen_labels = [] #assure only one example per subject is seen in test
        predictions = defaultdict(list)
        examples = 0

        with torch.no_grad():
            for i, data_test in enumerate(test_loader):
                anchor_test_id, anchor_test_visit = data_test['input_anchor']['id'][0]
                               
                if anchor_test_visit == 'visit2': continue  #ensure visit1 to visit2 test
                input_achor_test = data_test['input_anchor'].to(device)
                input_positive_test = data_test['input_positive'].to(device)
                input_negative_test = data_test['input_negative'].to(device)

                #Get pair similarity:
                out1_pos, out2_pos = model(input_achor_test,input_positive_test)
                disimilarity_positive = F.pairwise_distance(out1_pos, out2_pos)
                
                out1_neg, out2_neg = model(input_anchor,input_negative_test)
                disimilarity_negative = F.pairwise_distance(out1_neg, out2_neg)

                if disimilarity_positive < disimilarity_negative:
                    correct += 1

                examples += 1

                delta = disimilarity_negative - disimilarity_positive
                delta_loss[i].append(delta)
            
            mean_delta = torch.mean(torch.tensor([v[-1] for (k,v) in delta_loss.items()]))
            mean_delta_list.append(mean_delta)
            accuracy = correct/examples
            accuracy_list.append(accuracy)

            log = 'Epoch: {:03d}, train_+loss: {:.3f}, test_acc: {:.3f}, mean_delta: {:.4f}, lr: {:.2E}'
            print(log.format(e+1,np.mean(epoch_loss),accuracy,mean_delta,optimizer.param_groups[0]['lr']))

    # np.savez('outfile.npz', loss=training_losses,counter=counter, accuracy=accuracy_list, delta=mean_delta_list)
    # torch.save(model.state_dict(), checkpoint)

#Plots-----------------------------------------------------------------------------

    plt.plot(range(counter),np.mean(training_losses,axis=1), label='Training loss')
    plt.title('Training Loss') 
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
    plt.grid()
    plt.show()

    plt.plot(range(e+1),mean_delta_list)
    plt.title('Mean delta per epoch') 
    plt.xlabel('Epoch')
    plt.ylabel('Mean delta')
    plt.grid()
    plt.show()

