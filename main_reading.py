import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from torch.autograd import Variable
from dataset_loader import ACERTA_FP, ACERTA_reading
from torch import autograd
from models import Siamese_GeoChebyConv, Siamese_GeoSAGEConv, Siamese_GeoChebyConv_Read, GeoChebyConv
from utils import ContrastiveLoss
from torch.nn import BCEWithLogitsLoss
import sys
import argparse
from collections import defaultdict
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.stats import mode 
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import seaborn as sns

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:',device)
    
    checkpoint = 'checkpoints/'
    classification = 'reading'

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_type', type=str, default='PSC',
                        help='Type of feature data input', choices=['PSC','betas'])   
    parser.add_argument('--condition', type=str, default='all',
                        help='Task condition used as input', choices=['reg','irr','pse','all'])
    parser.add_argument('--split', type=float, default=0.7,
                        help='Size of training set')
    parser.add_argument('--adj_threshold', type=float, default='0.5',
                        help='Threshold for RST connectivity matrix edge selection')
    parser.add_argument('--model', type=str, default='gcn_single',
                        help='GCN model', choices=['gcn_cheby','gcn_cheby_bce','sage','gcn_single'])
    parser.add_argument('--hidden', type=int, default=16,
                        help='Number of hidden layers')
    parser.add_argument('--training_batch', type=int, default=4,
                        help='Training batch size')
    parser.add_argument('--test_batch', type=int, default=1,
                        help='Test batch size')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2,
                        help='Weight decay magnitude')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout magnitude')
    parser.add_argument('--loss_margin', type=float, default=0.2,
                        help='Margin for Contrastive Loss function')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--early_stop', type=int, default=99,
                        help='Epochs for early stop')
    parser.add_argument('--scheduler', type=bool, default=True,
                        help='Whether to use learning rate scheduler')
    parser.add_argument('--outfile', type=str, default='outfile.npz',
                        help='Path for output file containing results metrics.')
    args = parser.parse_args()


    training_set = ACERTA_reading(set_split='training', split=args.split, input_type=args.input_type,
                            condition=args.condition, adj_threshold=args.adj_threshold)
    test_set = ACERTA_reading(set_split='test', split=args.split, input_type=args.input_type,
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

    elif args.model == 'gcn_cheby_bce':
        model = Siamese_GeoChebyConv_Read(nfeat=nfeat,
                                     nhid=args.hidden,
                                     nclass=1,
                                     dropout=args.dropout)

    elif args.model == 'gcn_single':
        model = GeoChebyConv(nfeat=nfeat,
                            nhid=args.hidden,
                            nclass=1,
                            dropout=args.dropout)

    model.to(device)
    criterion = BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)
    if args.scheduler:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.5,patience=10,min_lr=1e-6)
        # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
        #                                 milestones=[5,30,50,80,150], gamma=0.5)
#Training-----------------------------------------------------------------------------

    counter=0
    training_losses = []
    test_losses = []
    accuracy_list = []
    for e in range(args.epochs):
        model.train()
        epoch_loss = []
        for i, data in enumerate(tqdm(train_loader)):

            if args.model == 'gcn_cheby_bce':
                input_anchor = data['input_anchor'].to(device)
                input_pair = data['input_pair'].to(device)
                label = data['label'].unsqueeze(1).to(device)
                
                #Match pair:
                output = model(input_anchor,input_pair).to(device)

            elif args.model == 'gcn_single':
                input_anchor = data['input_anchor'].to(device)
                input_pair = data['input_pair'].to(device)
                label = data['label_single'].to(device)

                output = model(input_anchor).to(device)

            training_loss = criterion(output, label.float())
            epoch_loss.append(training_loss.item())
            optimizer.zero_grad()
            training_loss.backward()
            optimizer.step()

        counter += 1
        training_losses.append(epoch_loss)
        if args.scheduler:
            lr_scheduler.step(np.mean(epoch_loss))
            # lr_scheduler.step()

#Testing-----------------------------------------------------------------------------
        model.eval()
        correct = 0
        predictions = defaultdict(list)
        y_prediction = []
        y_output = []
        y_true = []
        test_epoch_loss = []
        test_counter = 0
        with torch.no_grad():
            for i, data_test in enumerate(test_loader):
                anchor_test_id, anchor_test_visit = data_test['input_anchor']['id'][0]

                if args.model == 'gcn_cheby_bce':         
                    input_achor_test = data_test['input_anchor'].to(device)
                    input_pair_test = data_test['input_pair'].to(device)
                    label_test = data_test['label']
                    #Get pair prediction:
                    output = model(input_achor_test,input_pair_test)
                
                elif args.model == 'gcn_single':         
                    input_achor_test = data_test['input_anchor'].to(device)
                    label_test = data_test['label_single']
                    output = model(input_achor_test)

                test_loss = criterion(output, label_test.float().squeeze())
                #predict
                if nn.Sigmoid()(output)>0.5:
                    prediction = 1
                else:
                    prediction = 0

                if prediction == label_test:
                    correct += 1

                y_output.append(nn.Sigmoid()(output))
                y_prediction.append(prediction)
                y_true.append(label_test)
                test_epoch_loss.append(test_loss)
            
            test_losses.append(test_epoch_loss)
            # print('Pred: ',prediction)
            # print("Label: ", label_test)
            # print("Id Anchor: ", data_test['anchor_id'])
            # print("Id Pair: ", data_test['pair_id'])
            print("Correct: ",correct)
            accuracy = correct/len(test_loader)
            accuracy_list.append(accuracy)
            print("Predictions: ",y_prediction)

            log = 'Epoch: {:03d}, training_loss: {:.3f}, test_loss: {:.3f}, test_acc: {:.3f}, lr: {:.2E}'
            print(log.format(e+1,np.mean(epoch_loss),np.mean(test_epoch_loss),accuracy,optimizer.param_groups[0]['lr']))

    cm = confusion_matrix(y_true, y_prediction,normalize='true')
    fpr, tpr, thresholds = roc_curve(y_true, torch.tensor(y_output).cpu())
    auc_score = roc_auc_score(y_true, y_prediction)
    np.savez(args.outfile, training_loss=training_losses, test_loss=test_losses, counter=counter, accuracy=accuracy_list, \
            cm=cm,fpr=fpr,tpr=tpr,thresholds=thresholds,auc_score=auc_score,y_true=y_true,y_prediction=y_prediction)
    # torch.save(model.state_dict(), f"{checkpoint}chk_{classification}_{accuracy:.3f}.pth")

# run main_reading.py --model gcn_single --lr 5e-5 --epochs 200 --training_batch 4 --hidden 16 --condition ps
#    ...: e --split 0.8 --input_type PSC --adj_threshold 0.7 --dropout 0.6  

#Plots-----------------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(10,8))
    plt.plot(range(counter),np.mean(training_losses,axis=1), label='Training loss')
    plt.plot(range(counter),np.mean(test_losses,axis=1), label='Validation loss')
    plt.title('BCE Loss',fontsize=20)
    plt.xlabel('Epochs',fontsize=20)
    plt.ylabel('Loss',fontsize=20)
    plt.legend(prop={'size': 16})
    plt.grid()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    plt.show()

    fig, ax = plt.subplots(figsize=(10,8))
    plt.plot(range(e+1),accuracy_list)
    plt.title('Accuracy per epoch',fontsize=20)
    plt.xlabel('Epoch',fontsize=20)
    plt.ylabel('Accuracy',fontsize=20)
    plt.grid()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    plt.show()

    fig, ax = plt.subplots(figsize=(10,8))  
    sns.heatmap(cm, annot=True, ax = ax, fmt='.2g',cmap='Blues',annot_kws={"fontsize":18})  
    ax.set_xlabel('Predicted',fontsize=20)
    ax.set_ylabel('True',fontsize=20)
    ax.set_title('Confusion Matrix',fontsize=20)
    ax.xaxis.set_ticklabels(['Good', 'Bad'],fontsize=18); ax.yaxis.set_ticklabels(['Good', 'Bad'],fontsize=18)
    plt.show()

    fig, ax = plt.subplots(figsize=(10,8))  
    plt.plot(fpr,tpr, label='ROC curve (area = %0.2f)' % auc_score)
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
    plt.grid()
    plt.title('ROC Curve',fontsize=20)
    plt.xlabel('False Positive Rate',fontsize=20)
    plt.ylabel('True Positive Rate',fontsize=20)
    plt.legend(prop={'size': 16})
    plt.show()