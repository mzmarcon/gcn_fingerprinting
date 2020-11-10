import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.nn.functional as F
from st_dataset_loader import ACERTA_reading_ST
from models import *
from utils import ContrastiveLoss
import sys
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import seaborn as sns

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:',device)
    
    checkpoint = 'checkpoints/'
    classification = 'reading'

    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', type=str, default='all',
                        help='Task condition used as input', choices=['reg','irr','pse','all'])
    parser.add_argument('--split', type=float, default=0.9,
                        help='Size of training set')
    parser.add_argument('--adj_threshold', type=float, default='0.5',
                        help='Threshold for RST connectivity matrix edge selection')
    parser.add_argument('--model', type=str, default='gcn_single',
                        help='GCN model', choices=['gcn_cheby','gcn_cheby_bce','sage','gcn_single'])
    parser.add_argument('--training_batch', type=int, default=4,
                        help='Training batch size')
    parser.add_argument('--test_batch', type=int, default=2,
                        help='Test batch size')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3,
                        help='Weight decay magnitude')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout magnitude')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--no_scheduler', action='store_true',
                        help='Whether to use learning rate scheduler')
    parser.add_argument('--outfile', type=str, default='outfile.npz',
                        help='Path for output file containing results metrics.')
    args = parser.parse_args()

    #load and split dataset
    dataset = ACERTA_reading_ST()
    train_set_size = int(len(dataset)*args.split)
    val_set_size = len(dataset)-train_set_size
    train_set, test_set = random_split(dataset,[train_set_size,val_set_size])

    train_loader = DataLoader(train_set, shuffle=True, drop_last=True,
                                batch_size=args.training_batch)
    test_loader = DataLoader(test_set, shuffle=False, drop_last=False,
                                batch_size=args.test_batch)

    #TODO fix ADJ loading: use in processing function and TemporalModel loading.

    model = TemporalModel(1,1,None,True,dataset.adj_rst)
    model.to(device)
    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.75,patience=40,min_lr=1e-6)

#Training-----------------------------------------------------------------------------

    counter=0
    training_losses = []
    test_losses = []
    accuracy_list = []
    for e in range(args.epochs):
        model.train()
        epoch_loss = []
        for i, data in enumerate(tqdm(train_loader)):

            input_anchor = data['input_anchor'].unsqueeze(1).unsqueeze(4).to(device)
            label = data['label_single'].to(device)

            output = model(input_anchor).to(device)

            training_loss = criterion(output.squeeze(), label.float())
            epoch_loss.append(training_loss.item())
            optimizer.zero_grad()
            training_loss.backward()
            optimizer.step()

        counter += 1
        training_losses.append(epoch_loss)
        if not args.no_scheduler:
            lr_scheduler.step(np.mean(epoch_loss))
            # lr_scheduler.step()

#Testing-----------------------------------------------------------------------------
        model.eval()
        correct = 0
        y_prediction = []
        y_output = []
        y_true = []
        test_epoch_loss = []
        test_counter = 0
        ex_count = 0
        with torch.no_grad():
            for i, data_test in enumerate(test_loader):
                anchor_test_id, anchor_test_visit = map(list,data_test['anchor_info'])


                input_achor_test = data_test['input_anchor'].unsqueeze(1).unsqueeze(4).to(device)
                label_test = data_test['label_single'].to(device)
                output_test = model(input_achor_test)

                test_loss = criterion(output_test, label_test.unsqueeze(1).float())
                test_epoch_loss.append(test_loss)

                #predict
                for n,_ in enumerate(output_test):
                    if output_test[n]>0.5:
                        prediction = 1
                    else:
                        prediction = 0

                    if prediction == label_test[n]:
                        correct += 1
                    ex_count += 1
                    y_output.append(output_test[n].item())
                    y_prediction.append(prediction)
                    y_true.append(label_test[n].item())

            test_losses.append(test_epoch_loss)
            # print('Pred: ',prediction)
            # print("Label: ", label_test)
            # print("Id Anchor: ", data_test['anchor_id'])
            # print("Id Pair: ", data_test['pair_id'])
            print("Correct: {}/{}".format(correct,(len(test_loader)*args.test_batch)))
            accuracy = correct/(len(test_loader)*args.test_batch)
            accuracy_list.append(accuracy)
            print("Predictions: {} - len: {}".format(y_prediction,len(y_prediction)))
            print("True: {}".format(y_true))        
            print(', '.join('{:.3f}'.format(f) for f in y_output))
            print(np.unique(y_prediction,return_counts=True))

            log = 'Epoch: {:03d}, training_loss: {:.3f}, test_loss: {:.3f}, test_acc: {:.3f}, lr: {:.2E}'
            print(log.format(e+1,np.mean(epoch_loss),torch.mean(torch.tensor(test_epoch_loss)),accuracy,optimizer.param_groups[0]['lr']))

    np.savez('results.npz',train_loss=training_losses,test_loss=test_losses,test_acc=accuracy_list)
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
