import numpy as np
import torch
import os
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
from st_dataset_loader import ACERTA_regression_ST
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
    parser.add_argument('--task', type=str, default='regression',
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
    parser.add_argument('--factor', type=float, default=0.6,
                        help='Factor for scheduler updates.')
    parser.add_argument('--outfile', type=str, default='outfile',
                        help='Name of output file containing results metrics.')
    parser.add_argument('--edgefile', type=str, default='edge_imp',
                        help='Name of output file containing edge importance.')
    parser.add_argument('--minmax', action='store_true',
                        help='Whether to use MinMax normalization')
    parser.add_argument('--window_t', type=int, default=300,
                        help='Window size for timeseries augmentation.')
    parser.add_argument('--binary_adj', action='store_true',
                        help='Wether to use unweighted or "binary" adjacency.')
    args = parser.parse_args()

    # load dataset
    if args.task == 'regression':
        dataset = ACERTA_regression_ST(args.split,args.condition,args.adj_threshold,args.window_t,args.minmax,args.binary_adj)
        output_path = 'output/dyslexia/'
        checkpoint = 'checkpoints/dyslexia/'

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

    model = TemporalRegressionModel(1,1,None,True,dataset.adj_matrix,args.dropout)
    model.to(device)

    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=args.factor,patience=args.patience,min_lr=1e-6)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,50,70,80,90], gamma=0.5, last_epoch=-1)

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

            training_loss = criterion(output, label.float().view(-1,1))
            epoch_loss.append(training_loss.item())
            optimizer.zero_grad()
            training_loss.backward()
            optimizer.step()
                
        # if e % 99 == 0 and e>0:
        if e == args.epochs-1:
            file_list = os.listdir(output_path+'edge_importance/')
            edge_imp_id = np.max([int(item.split('.')[-2].split('_')[-1]) for item in file_list if not 'csv' in item]) + 1
            for importance in model.edge_importance:
                edge_importances = importance*importance+torch.transpose(importance*importance,0,1)
                edge_imp = torch.squeeze(edge_importances.data).cpu().numpy()
                filename = output_path + "edge_importance/" + args.edgefile + "_" + str(edge_imp_id)
                np.save(filename, edge_imp)

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
        test_ids = []
        with torch.no_grad():
            for i, data_test in enumerate(test_loader):
                anchor_test_id, anchor_test_visit = map(list,data_test['anchor_info'])

                test_ids.extend(data_test['anchor_info'][0][:])

                input_achor_test = data_test['input_anchor'].unsqueeze(1).unsqueeze(4).to(device)
                label_test = data_test['label_single'].to(device)
                output_test = model(input_achor_test)

                test_loss = criterion(output_test, label_test.unsqueeze(1).float().view(-1,1))
                test_epoch_loss.append(test_loss)

            test_losses.append(test_epoch_loss)
            print('Output: ',output_test)
            print('Label: ', label_test.view(-1,1))

            log = 'Epoch: {:03d}, train_loss: {:.3f}, test_loss: {:.3f}, lr: {:.2E}'
            print(log.format(e+1,np.mean(epoch_loss),torch.mean(torch.tensor(test_epoch_loss)),optimizer.param_groups[0]['lr']))
            
    outfile_id = len([file for file in os.listdir(output_path) if 'outfile' in file]) + 1
    outfile_name = output_path + args.outfile + '_' + str(outfile_id)
    checkpoint_id = len(os.listdir(checkpoint)) + 1

    np.savez(outfile_name, training_loss=training_losses, test_loss=test_losses, args=args)
    # torch.save(model.state_dict(), f"{checkpoint}chk_ST_{args.task}_{checkpoint_id}.pth")

