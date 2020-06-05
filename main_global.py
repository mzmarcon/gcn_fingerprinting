import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from torch.autograd import Variable
from dataset_loader import ACERTA_FP
from torch import autograd
from models import Siamese_GeoChebyConv, GeoSAGEConv, Siamese_GeoSAGEConv,  Siamese_GlobalCheby
from utils import ContrastiveLoss, GlobalLoss
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
    
    params = { 'model': 'gcn_global_cheby',
               'train_batch_size': 1,
               'test_batch_size': 1,
               'learning_rate': 2.5e-4,
               'weight_decay': 1e-1,
               'epochs': 120,
               'early_stop': 10,
               'dropout': 0.5,
               'voting_examples': 1}
    
    training_set = ACERTA_FP(set='training', split=0.8, type='condition')
    test_set = ACERTA_FP(set='test', split=0.8, type='condition')
    
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

    if params['model'] == 'gcn_global_cheby':
        model = Siamese_GlobalCheby(nfeat=nfeat,
                            nhid=32,
                            nclass=1,
                            dropout=params['dropout'])

    if params['model'] == 'gcn_cheby':
        model = Siamese_GeoChebyConv(nfeat=nfeat,
                                     nhid=32,
                                     nclass=1,
                                     dropout=params['dropout'])
    model.to(device)
    
    criterion = GlobalLoss(margin=0.3)

    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'],
                                weight_decay=params['weight_decay'])

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                             milestones=[30,80,100,450,1000,1500], gamma=0.5)

#Training-----------------------------------------------------------------------------

    counter=0
    delta_loss = defaultdict(list)
    accuracy_list = []
    mean_delta_list =[]
    global_losses = []
    for e in range(params['epochs']):
        model.train()
        positive_similarities = []
        negative_similarities = []
        for i, data in enumerate(tqdm(train_loader)):
            input_anchor = data['input_anchor'].to(device)

            _, anchor_visit = data['input_anchor']['id'][0]
            if anchor_visit == 'visit2': continue  #ensure visit1 to visit2

            input_positive = data['input_positive'].to(device)
            input_negative = data['input_negative'].to(device)
            sys.exit()
            #Positive pair:
            similarity_positive = model(input_anchor,input_positive)
            positive_similarities.append(similarity_positive)
            #Negative pair:
            similarity_negative = model(input_anchor,input_negative)
            negative_similarities.append(similarity_negative)

        global_loss = criterion(positive_similarities, negative_similarities)
        global_losses.append(global_loss.item())
        optimizer.zero_grad()
        global_loss.backward()
        optimizer.step()

        counter += 1
        lr_scheduler.step()

#Testing-----------------------------------------------------------------------------
        model.eval()
        correct=0
        seen_labels = [] #assure only one example per subject is seen in test
        predictions = defaultdict(list)

        with torch.no_grad():
            for i, data_test in enumerate(test_loader):
                anchor_test_id, anchor_test_visit = data_test['input_anchor']['id'][0]
                anchor_test_matching = data_test['matching_idx']
                               
                if anchor_test_visit == 'visit2': continue  #ensure visit1 to visit2 test
                input_achor_test = data_test['input_anchor'].to(device)

                label = anchor_test_id  #test only N examples per subject
                if not seen_labels.count(label)>=params['voting_examples']:
                    seen_labels.append(label)
                else: 
                    continue 

                similarities = defaultdict(list)

                seen_labels_test = []
                for n, data_test_example in enumerate(test_loader): #Compare example to each example in the test set:
                    example_test_id, example_test_visit = data_test_example['input_anchor']['id'][0]

                    if example_test_visit == 'visit1': continue #ensure visit1 to visit2 test
                    input_example_test = data_test_example['input_anchor'].to(device)

                    #Get pair similarity:
                    output = model(input_achor_test,input_example_test)
                    similarity = output
                    similarities[example_test_id].append(similarity)

                mean_similarities = defaultdict(list)  #get mean similaritie for the 20 examples analyzed per subject
                for k,v in similarities.items():
                    mean_similarities[k]= torch.mean(torch.stack(v))
                prediction = min(mean_similarities, key=similarities.get) #make prediction
                predictions[anchor_test_id].append(prediction)

                if len(predictions[anchor_test_id])==params['voting_examples']: #get most common prediction from voting list
                    voting_prediction = mode(predictions[anchor_test_id])[0][0]                     
                    if voting_prediction == label:
                        correct = correct+1

                    delta = torch.abs(mean_similarities[voting_prediction]-mean_similarities[label]) #compute delta between prediction and true
                    print("Diff delta: ",delta)
                    delta_loss[i].append(delta)
            
            mean_delta = torch.mean(torch.tensor([v[-1] for (k,v) in delta_loss.items()]))
            mean_delta_list.append(mean_delta)
            accuracy = correct/len(predictions)
            accuracy_list.append(accuracy)

            log = 'Epoch: {:03d}, train_loss: {:.3f}, test_acc: {:.3f}, mean_delta: {:.4f}, lr: {:.2E}'
            print(log.format(e+1,global_loss,accuracy,mean_delta,optimizer.param_groups[0]['lr']))


#Plots-----------------------------------------------------------------------------

    plt.plot(range(counter),positive_losses, label='Positive loss')
    plt.plot(range(counter),negative_losses, label='Negative loss')
    plt.title('Positive vs Negative Loss') 
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    np.savez('outfile.npz', positive=positive_losses, negative=negative_losses, counter=counter, accuracy=accuracy_list, delta=mean_delta_list)

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

    # torch.save(model.state_dict(), checkpoint)
