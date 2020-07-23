import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from torch.autograd import Variable
from dataset_loader import ACERTA_FP
from torch.nn import BCEWithLogitsLoss
from torch import autograd
from models import Siamese_GeoChebyConv, GeoSAGEConv, Siamese_GeoSAGEConv, Siamese_GeoChebyConv_Read
from utils import ContrastiveLoss
import sys
import argparse
from collections import defaultdict
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.stats import mode 
from tqdm import tqdm

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:',device)
    
    checkpoint = 'checkpoints/'
    classification = 'FP'

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_type', type=str, default='PSC',
                        help='Type of feature data input', choices=['PSC','betas'])   
    parser.add_argument('--condition', type=str, default='all',
                        help='Task condition used as input', choices=['reg','irr','pse','all'])
    parser.add_argument('--split', type=float, default=0.8,
                        help='Size of training set')
    parser.add_argument('--adj_threshold', type=float, default='0.5',
                        help='Threshold for RST connectivity matrix edge selection')
    parser.add_argument('--model', type=str, default='gcn_cheby_bce',
                        help='GCN model', choices=['gcn_cheby','gcn_cheby_bce','sage'])
    parser.add_argument('--hidden', type=int, default=16,
                        help='Number of hidden layers')
    parser.add_argument('--training_batch', type=int, default=8,
                        help='Training batch size')
    parser.add_argument('--test_batch', type=int, default=1,
                        help='Test batch size')
    parser.add_argument('--lr', type=float, default=1e-6,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-1,
                        help='Weight decay magnitude')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout magnitude')
    parser.add_argument('--loss_margin', type=float, default=0.2,
                        help='Margin for Contrastive Loss function')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs')
    parser.add_argument('--early_stop', type=int, default=99,
                        help='Epochs for early stop')
    parser.add_argument('--scheduler', type=bool, default=False,
                        help='Whether to use learning rate scheduler')
    args = parser.parse_args()


    training_set = ACERTA_FP(set_split='training', split=args.split, input_type=args.input_type,
                            condition=args.condition, adj_threshold=args.adj_threshold)
    test_set = ACERTA_FP(set_split='test', split=args.split, input_type=args.input_type,
                            condition=args.condition, adj_threshold=args.adj_threshold)
    
    train_loader = DataLoader(training_set, shuffle=True, drop_last=True,
                                batch_size=args.training_batch)
    test_loader = DataLoader(test_set, shuffle=False, drop_last=False,
                                batch_size=args.test_batch)
    

    nfeat = train_loader.__iter__().__next__()['input_anchor']['x'].shape[1]
    print("NFEAT: ",nfeat)
    
    if args.model == 'sage':
        model = Siamese_GeoSAGEConv(nfeat=nfeat,
                            nhid=args.hidden,
                            nclass=1,
                            dropout=args.dropout)
        criterion = ContrastiveLoss(args.loss_margin)

    elif args.model == 'gcn_cheby':
        model = Siamese_GeoChebyConv(nfeat=nfeat,
                                     nhid=args.hidden,
                                     nclass=1,
                                     dropout=args.dropout)
        criterion = ContrastiveLoss(args.loss_margin)

    elif args.model == 'gcn_cheby_bce':
        model = Siamese_GeoChebyConv_Read(nfeat=nfeat,
                                     nhid=args.hidden,
                                     nclass=1,
                                     dropout=args.dropout)
        criterion = BCEWithLogitsLoss()
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)
    if args.scheduler:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                milestones=[20,60,100,150,450,1000,1500], gamma=0.5)

#Training-----------------------------------------------------------------------------

    delta_loss = defaultdict(list)
    accuracy_list = []
    training_losses = []
    test_losses = []
    counter=0   

    for e in range(args.epochs):
        model.train()
        epoch_loss = []
        for i, data in enumerate(tqdm(train_loader)):
            input_anchor = data['input_anchor'].to(device)
            input_pair = data['input_pair'].to(device)
            label = data['label'].to(device)
            label = torch.split(label,input_anchor.num_graphs)

            if args.model == 'gcn_cheby':
                #Match pair:
                out1, out2 = model(input_anchor,input_pair)
                training_loss = criterion(out1, out2, label)

            elif args.model == 'gcn_cheby_bce':
                #Match pair:
                output = model(input_anchor,input_pair)
                training_loss = criterion(output, label[0].unsqueeze(1).float())

            epoch_loss.append(training_loss.item())
            optimizer.zero_grad()
            training_loss.backward()
            optimizer.step()

        counter += 1
        training_losses.append(epoch_loss)
        if args.scheduler:
            lr_scheduler.step()

#Testing-----------------------------------------------------------------------------
        model.eval()
        correct=0
        examples = 0
        predictions = defaultdict(list)
        y_true = []
        y_prediction = []
        test_epoch_loss = []

        with torch.no_grad():
            for i, data_test in enumerate(test_loader):
                anchor_test_id, anchor_test_visit = data_test['input_anchor']['id'][0]
                               
                if anchor_test_visit == 'visit2': continue  #ensure visit1 to visit2 test
                input_achor_test = data_test['input_anchor'].to(device)

                if args.model == 'gcn_cheby':
                    input_positive_test = data_test['input_positive'].to(device)
                    input_negative_test = data_test['input_negative'].to(device)

                    #compute pair similarities:
                    out1_pos, out2_pos = model(input_achor_test,input_positive_test)
                    disimilarity_positive = torch.sum(F.pairwise_distance(out1_pos, out2_pos))
                    
                    out1_neg, out2_neg = model(input_achor_test,input_negative_test)
                    disimilarity_negative = torch.sum(F.pairwise_distance(out1_neg, out2_neg))

                    if disimilarity_positive < disimilarity_negative:
                        correct += 1

                    test_loss_pos = criterion(out1_pos, out2_pos, [[0]])
                    test_loss_neg = criterion(out1_neg, out2_neg, [[1]])
                    test_loss = torch.mean(torch.stack((test_loss_pos,test_loss_neg)))


                elif args.model == 'gcn_cheby_bce':
                    input_pair_test = data_test['input_pair'].to(device)
                    label_test = data_test['label']

                    #get pair prediction:
                    output = model(input_achor_test,input_pair_test)
                    test_loss = criterion(output.squeeze(1), label_test.to(device).float())

                    #predict
                    if nn.Sigmoid()(output)>0.5:
                        prediction = 1
                    else:
                        prediction = 0
    
                    if prediction == label_test:
                        correct += 1

                    y_prediction.append(prediction)
                    y_true.append(label_test)

                examples += 1
                test_epoch_loss.append(test_loss.item())
            accuracy = correct/examples
            accuracy_list.append(accuracy)
            test_losses.append(test_epoch_loss)

            log = 'Epoch: {:03d}, training_loss: {:.3f}, test_loss: {:.3f}, test_acc: {:.3f}, lr: {:.2E}'
            print(log.format(e+1,np.mean(epoch_loss),np.mean(test_epoch_loss),accuracy,optimizer.param_groups[0]['lr']))

    # fpr, tpr, thresholds = roc_curve(y_true, y_prediction)
    np.savez('outfile.npz', training_loss=training_losses,test_losses=test_losses,counter=counter, accuracy=accuracy_list, y_true=y_true, y_prediction=y_prediction)
    # torch.save(model.state_dict(), f"{checkpoint}chk_{classification}_{args.condition}_{args.adj_threshold}_{accuracy:.3f}.pth")

#Plots-----------------------------------------------------------------------------

    #plot training loss  
    fig = plt.figure(figsize=(10,8)) 
    plt.plot(range(counter),np.mean(training_losses,axis=1), label='Training loss') 
    plt.plot(range(counter),np.mean(test_losses,axis=1), label='Validation loss') 
    plt.title('Fingerprinting',fontsize=20) 
    plt.xlabel('Epochs',fontsize=20) 
    plt.ylabel('Loss',fontsize=20) 
    plt.legend(prop={'size': 16}) 
    # plt.grid()
    plt.show()

    #plot accuracy over epochs
    fig = plt.figure(figsize=(10,8)) 
    plt.plot(range(e+1),accuracy_list)
    plt.title('Fingerprinting',fontsize=20) 
    plt.xlabel('Epoch',fontsize=20)
    plt.ylabel('Accuracy',fontsize=20)
    # plt.grid()
    plt.show()


