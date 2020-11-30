import numpy as np
import torch
import os
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
from st_dataset_loader import ACERTA_reading_ST, ACERTA_dyslexic_ST
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
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='dyslexia',
                        help='Task type', choices=['dyslexia','reading'])
    parser.add_argument('--condition', type=str, default='none',
                        help='Task condition used as input', choices=['reg','irr','pse','all','none'])
    parser.add_argument('--split', type=float, default=0.7,
                        help='Size of training set')
    parser.add_argument('--adj_threshold', type=float, default='0.5',
                        help='Threshold for RST connectivity matrix edge selection')
    parser.add_argument('--model', type=str, default='gcn_single',
                        help='GCN model', choices=['gcn_cheby','gcn_cheby_bce','sage','gcn_single'])
    parser.add_argument('--training_batch', type=int, default=4,
                        help='Training batch size')
    parser.add_argument('--test_batch', type=int, default=2,
                        help='Test batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2,
                        help='Weight decay magnitude')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout magnitude')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--no_scheduler', action='store_true',
                        help='Whether to use learning rate scheduler')
    parser.add_argument('--patience', type=int, default=10,
                        help='Scheduler patience in epochs.')
    parser.add_argument('--outfile', type=str, default='outfile',
                        help='Name of output file containing results metrics.')
    parser.add_argument('--prune', action='store_true',
                        help='Whether to prune out cerebellum.')
    parser.add_argument('--window_t', type=int, default=300,
                        help='Window size for timeseries augmentation.')
    args = parser.parse_args()

    # load dataset
    if args.task == 'dyslexia':
        dataset = ACERTA_dyslexic_ST(args.split,args.condition,args.adj_threshold,args.window_t,args.prune)
        output_path = 'output/dyslexia/'
        checkpoint = 'checkpoints/dyslexia/'

    elif args.task == 'reading':
        dataset = ACERTA_reading_ST(args.split,args.condition,args.adj_threshold,args.window_t,args.prune)
        output_path = 'output/reading/'
        checkpoint = 'checkpoints/reading/'
    
    # load split indices
    train_idx = dataset.train_idx
    test_idx = dataset.test_idx

    np.random.shuffle(train_idx)
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    train_loader = DataLoader(dataset, drop_last=True,
                                batch_size=args.training_batch, sampler=train_sampler)
    test_loader = DataLoader(dataset, drop_last=False,
                                batch_size=args.test_batch, sampler=test_sampler)

    model = TemporalModel(1,1,None,True,dataset.adj_matrix,args.dropout)
    model.to(device)
    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.6,patience=args.patience,min_lr=1e-6)

#Training-----------------------------------------------------------------------------

    counter=0
    training_losses = []
    test_losses = []
    train_accuracy_list = []
    accuracy_list = []

    for e in range(args.epochs):
        model.train()
        epoch_loss = []
        correct = 0
        ex_count_train = 0
        for i, data in enumerate(tqdm(train_loader)):

            input_anchor = data['input_anchor'].unsqueeze(1).unsqueeze(4).to(device)
            label = data['label_single'].to(device)

            output = model(input_anchor).to(device)

            training_loss = criterion(output.squeeze(), label.float())
            epoch_loss.append(training_loss.item())
            optimizer.zero_grad()
            training_loss.backward()
            optimizer.step()
                
            # Make predictions every 5 mini-batches
            if i % 5 == 0:
                for n,_ in enumerate(output):
                    if output[n]>0.5:
                        prediction = 1
                    else:
                        prediction = 0
                    if prediction == label[n]:
                        correct += 1
                    ex_count_train += 1

        if e % 99 == 0 and e>0:
            edge_imp_id = len(os.listdir(output_path+'edge_importance/')) + 1
            for importance in model.edge_importance:
                edge_importances = importance*importance+torch.transpose(importance*importance,0,1)
                edge_imp = torch.squeeze(edge_importances.data).cpu().numpy()
                filename = output_path + "edge_importance/edge_imp_" + str(edge_imp_id)
                np.save(filename, edge_imp)

        counter += 1
        train_accuracy = correct/ex_count_train
        train_accuracy_list.append(train_accuracy)
        training_losses.append(epoch_loss)
        if not args.no_scheduler:
            lr_scheduler.step(np.mean(epoch_loss))

#Testing-----------------------------------------------------------------------------
        model.eval()
        correct = 0
        y_prediction = []
        y_output = []
        y_true = []
        test_epoch_loss = []
        test_counter = 0
        ex_count = 0
        test_ids = []
        with torch.no_grad():
            for i, data_test in enumerate(test_loader):
                anchor_test_id, anchor_test_visit = map(list,data_test['anchor_info'])

                test_ids.extend(data_test['anchor_info'][0][:])

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
            # print("Predictions: {} - len: {}".format(y_prediction,len(y_prediction)))
            # print("True: {}".format(y_true))
            u,c = np.unique(y_true,return_counts=True)
            print("Chance: {}/{}={:.3f}".format(c[0],c[1],(c[0]/(c[0]+c[1]))))
            # print(', '.join('{:.3f}'.format(f) for f in y_output))
            print(np.unique(y_prediction,return_counts=True))

            log = 'Epoch: {:03d}, train_loss: {:.3f}, test_loss: {:.3f}, train_acc: {:.3f}, test_acc: {:.3f}, lr: {:.2E}'
            print(log.format(e+1,np.mean(epoch_loss),torch.mean(torch.tensor(test_epoch_loss)),train_accuracy,accuracy,optimizer.param_groups[0]['lr']))
            
    cm = confusion_matrix(y_true, y_prediction,normalize='true')
    fpr, tpr, thresholds = roc_curve(y_true, torch.tensor(y_output).cpu())
    auc_score = roc_auc_score(y_true, y_prediction)

    outfile_id = len([file for file in os.listdir(output_path) if 'outfile' in file]) + 1
    outfile_name = output_path + args.outfile + '_' + str(outfile_id)
    checkpoint_id = len(os.listdir(checkpoint)) + 1

    np.savez(outfile_name, training_loss=training_losses, test_loss=test_losses, counter=counter, accuracy=accuracy_list, train_accuracy=train_accuracy_list,\
            cm=cm,fpr=fpr,tpr=tpr,thresholds=thresholds,auc_score=auc_score,y_true=y_true,y_prediction=y_prediction)
    torch.save(model.state_dict(), f"{checkpoint}chk_ST_{args.task}_{checkpoint_id}.pth")

