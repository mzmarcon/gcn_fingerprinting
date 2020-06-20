import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from torch.autograd import Variable
from dataset_loader import ACERTA_FP, ACERTA_reading
from torch import autograd
from models import Siamese_GeoChebyConv, GeoSAGEConv, Siamese_GeoChebyConv_Read
from utils import ContrastiveLoss
from torch.nn import BCEWithLogitsLoss
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
               'learning_rate': 1e-5,
               'weight_decay': 1e-1,
               'epochs': 40,
               'early_stop': 10,
               'dropout': 0.5,
               'loss_margin': 0.2,
               'input_type': 'condition', #if 'condition', input are betas for condition. if 'allbetas', input vector with all betas.
               'condition': 'irr', #set type of input condition. 'irr', 'pse', 'reg' or 'all'.
               'adj_threshold': 0.5,
               'voting_examples': 1}


    training_set = ACERTA_reading(set_split='training', split=0.8, input_type=params['input_type'],
                            condition=params['condition'], adj_threshold=params['adj_threshold'])

    test_set = ACERTA_reading(set_split='test', split=0.8, input_type=params['input_type'],
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
        model = Siamese_GeoChebyConv_Read(nfeat=nfeat,
                                     nhid=32,
                                     nclass=1,
                                     dropout=params['dropout'])
    
    model.to(device)
    
    # criterion = ContrastiveLoss(params['loss_margin'])
    criterion = BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'],
                                weight_decay=params['weight_decay'])

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                             milestones=[20,150,200,450,1000,1500], gamma=0.5)

#Training-----------------------------------------------------------------------------

    counter=0
    training_losses = []
    delta_loss = defaultdict(list)
    accuracy_list = []
    mean_delta_list =[]

    for e in range(params['epochs']):
        model.train()
        epoch_loss = []
        for i, data in enumerate(tqdm(train_loader)):
            input_anchor = data['input_anchor'].to(device)
            input_pair = data['input_pair'].to(device)
            label = data['label'].unsqueeze(1).to(device)
            
            #Match pair:
            output = model(input_anchor,input_pair).to(device)

            training_loss = criterion(output, label.float())
            epoch_loss.append(training_loss.item())
            optimizer.zero_grad()
            training_loss.backward()
            optimizer.step()

        counter += 1
        training_losses.append(epoch_loss)
        # lr_scheduler.step()

#Testing-----------------------------------------------------------------------------
        model.eval()
        correct=0
        seen_labels = [] #assure only one example per subject is seen in test
        predictions = defaultdict(list)

        with torch.no_grad():
            for i, data_test in enumerate(test_loader):
                anchor_test_id, anchor_test_visit = data_test['input_anchor']['id'][0]
                               
                input_achor_test = data_test['input_anchor'].to(device)
                input_pair_test = data_test['input_pair'].to(device)
                label_test = data_test['label']
                #Get pair prediction:
                output = model(input_achor_test,input_pair_test)
                
                #predict
                if nn.Sigmoid()(output)>0.5:
                    prediction = 1
                else:
                    prediction = 0

                if prediction == label_test:
                    correct += 1
            
            print('Pred: ',prediction)
            print("Label: ", label_test)
            print("Id Anchor: ", data_test['anchor_id'])
            print("Id Pair: ", data_test['pair_id'])
            print("Correct: ",correct)
            accuracy = correct/len(test_loader)
            accuracy_list.append(accuracy)

            log = 'Epoch: {:03d}, train_loss: {:.3f}, test_acc: {:.3f}, lr: {:.2E}'
            print(log.format(e+1,np.mean(training_losses),accuracy,optimizer.param_groups[0]['lr']))

    np.savez('outfile.npz', loss=training_losses, counter=counter, accuracy=accuracy_list, delta=mean_delta_list)

    # torch.save(model.state_dict(), checkpoint)

#Plots-----------------------------------------------------------------------------

    plt.plot(range(counter),np.mean(training_losses,axis=1), label='Training loss')
    plt.title('Training Loss') 
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.plot(range(e+1),accuracy_list)
    plt.title('Accuracy per epoch') 
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.show()

