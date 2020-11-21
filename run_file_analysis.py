import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_fscore_support
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import sys
from nilearn import plotting
import itertools
import os
import pandas as pd
import copy

def get_max_weights(matrix,percentual):
    """
    Function that returns sorted "size" max weights and corresponding indices from a matrix

    returns: indices: coords of max_weights 
             weights: max_weights
             max_matrix: matrix on the shape of input matrix with max_values only 
    """
    indices = []
    weights = []
    total = np.unique(matrix)
    size = int(percentual * len(total))
    matrix_copy = copy.deepcopy(matrix)
    for n in range(size):
        max_value = np.max(matrix_copy)
        index = np.where(matrix_copy==max_value) 
        weights.append(max_value)
        indices.append(index)
        matrix_copy[index] = 0 

    max_matrix = np.zeros([matrix.shape[0],matrix.shape[0]])
    for n,_ in enumerate(indices):
        max_matrix[indices[n][0][0],indices[n][1][0]] = weights[n]

    return indices, weights, max_matrix

def plot_weights_connectome(weights_mask,threshold='90%',size=500,title='',colorbar=True,type='None'):
    
    #plot connectome with atlas coords
    if os.path.isfile('data/atlas_coords.npz'):
        atlas = np.load('data/atlas_coords.npz')
        coords = atlas['data'][:]
    else:
        print("Coords file not found.")
        raise FileNotFoundError()

    column_sum = np.sum(weights_mask,axis=1)
    column_scaled = (column_sum - column_sum.min(axis=0)) / (column_sum.max(axis=0) - column_sum.min(axis=0)) 
    column_scaled = column_scaled * (size - 5) + 5

    triu = np.triu(weights_mask)
    #mirror upper triangle to lower in order to plot
    fill_triu = triu + triu.T - np.diag(np.diag(triu))

    if type == 'neg':
        edge_cmap = 'Blues'
    else:
        edge_cmap = 'red_transparent'

    plotting.plot_connectome(fill_triu,coords,edge_threshold=threshold,node_size=column_scaled, annotate=True,
                            edge_vmin=fill_triu.min(),edge_vmax=fill_triu.max(),title=title,edge_cmap=edge_cmap,colorbar=colorbar,display_mode='lyrz')

    plotting.show()

def get_macro_matrix(matrix,csv_path='data/shen_268_parcellation_networklabels.csv',type='delta'):
    """
    Function that gets nodes for each macro region. Macro matrix contains delta between number of 
    positive and negative edges.

    returns: d: dict with nodes per region
             macro_matrix: matrix with positive edges - number negative edges
    """
    regions = pd.read_csv(csv_path)
    nodes = regions.Node.values
    networks = regions.Network.values
    network_n = len(np.unique(networks))

    # Make a dict with the nodes for each macro region
    d = {}
    # Get indices of each region in networks list and retrieve nodes of those indices in nodes list 
    for item in np.unique(networks): 
        d[item-1] = np.where(networks==item) # node index - 1 = network index (csv nodes indexed at 1)

    # Create macro matrix with delta between positive and negative values in each region.
    macro_matrix = np.zeros([network_n,network_n])
    
    if type == 'delta':
        for n in range(network_n):
            positive_n = len(matrix[d[n]][matrix[d[n]]>0]) #positive node values for macro region n
            negative_n = len(matrix[d[n]][matrix[d[n]]<0]) #negative node values for macro region n
            delta = positive_n - negative_n
            macro_matrix[n,n] = delta
    
    elif type == 'sum':
        for n in range(network_n):
            sum_ = np.sum(matrix[d[n]])
            macro_matrix[n,n] = sum_
    
    # Fill matrix
    perm = list(itertools.permutations(range(network_n),2))
    for item in perm:
        a = item[0] 
        b = item[1] 
        result_ab =  macro_matrix[a,a] + macro_matrix[b,b] 
        macro_matrix[a,b] = result_ab

    return d, macro_matrix

def plot_macro_matrix(macro_matrix,title=''):

    mask = np.triu(macro_matrix,1)
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    x_labels = ['Medial Frontal','Frontoparietal', 'Default Mode', 'Subcortical-Cerebellum',
                     'Motor','Visual I','Visual II', 'Visual Association']
    y_labels = ['Medial Frontal','Frontoparietal', 'Default Mode', 'Subcortical-Cerebellum',
                     'Motor','Visual I','Visual II', 'Visual Association']

    fig, ax = plt.subplots(figsize=(15,10))
    ax = sns.heatmap(np.tril(macro_matrix), mask=mask, cmap=cmap, center=0, xticklabels=x_labels,
                yticklabels=y_labels, square=True, linewidths=.5)

    ax.set_xticklabels(x_labels,rotation=20,ha='right',rotation_mode='anchor')
    ax.set_yticklabels(y_labels,rotation=15)
    # plt.title('(No positive edges - No negative edges) per region - '+title)
    plt.title(title)
    plt.show()


if __name__ == '__main__':
 
    file_path = sys.argv[1]
 
    if 'edge' in file_path:
        mode='edge'
    elif 'outfile' in file_path:
        mode='results'
    else:
        raise ValueError('Invalid analysis mode in argv[1] - edge or results')

    if 'dyslexia' in file_path:
        task = 'Dyslexia'
    elif 'reading' in file_path:
        task = 'Reading'

    print("Plotting {} anlysis for {} task.".format(mode,task))

    train_acc = False
    if mode == 'results':
        file = np.load(file_path,allow_pickle=True)
        fpr = file['fpr']
        tpr = file['tpr']
        cm = file['cm']
        auc = file['auc_score']
        acc_list = file['accuracy']
        training_loss = file['training_loss']
        test_loss = file['test_loss']
        y_true = file['y_true']
        y_prediction = file['y_prediction']
        counter = file['counter']
        if 'train_accuracy' in list(file.keys()):
                train_acc_list = file['train_accuracy']
                train_acc = True

        precision, recall, fscore, _ = precision_recall_fscore_support(list(y_true), 
                                                                       list(y_prediction),average='micro') 

        ### Print Summary ###
        print("Results summary:")
        print("Final training loss: {:.3f} | Final test loss: {:.3f}".format(np.mean(training_loss[-1]),torch.mean(torch.Tensor(list(test_loss[-1])))))
        print("Final test accuracy: {:.3f} | Top test accuracy: {:.3f}".format(acc_list[-1],acc_list.max()))
        print("AUC: {} | Precision: {} | Recall: {} | F-score: {}".format(auc,precision,recall,fscore))
        print("Confusion Matrix:\n",cm)

        ### Plots ###

        fig, ax = plt.subplots(figsize=(10,8))
        plt.plot(range(counter),np.mean(training_loss,axis=1), label='Training loss')
        plt.plot(range(counter),np.mean(test_loss,axis=1), label='Validation loss')
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
        if train_acc:
            plt.plot(train_acc_list,label="Train accuracy")
        plt.plot(acc_list,label='Validation accuracy')
        plt.title('Accuracy per epoch',fontsize=20)
        plt.xlabel('Epoch',fontsize=20)
        plt.ylabel('Accuracy',fontsize=20)
        plt.legend(prop={'size': 16})
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
        if task == 'Dyslexia':
            ax.xaxis.set_ticklabels(['Dyslexic', 'Control'],fontsize=18); ax.yaxis.set_ticklabels(['Dyslexic', 'Control'],fontsize=18)
        else:
            ax.xaxis.set_ticklabels(['Good', 'Bad'],fontsize=18); ax.yaxis.set_ticklabels(['Good', 'Bad'],fontsize=18)

        plt.show()

        fig, ax = plt.subplots(figsize=(10,8))  
        plt.plot(fpr,tpr, label='ROC curve (area = %0.2f)' % auc)
        plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
        plt.grid()
        plt.title('ROC Curve',fontsize=20)
        plt.xlabel('False Positive Rate',fontsize=20)
        plt.ylabel('True Positive Rate',fontsize=20)
        plt.legend(prop={'size': 16})
        plt.show()


    elif mode == 'edge':
        edge = np.load(file_path)
        file_name = file_path.split('/')[-1].split('.')[0]
       
        edge_clip = edge * (edge>edge.min())
        plot_weights_connectome(edge_clip,threshold='99.9%',size=200,title='Edge Importance ' + task + ': ' + file_name + ' - Zero Clipped')

        d_edge_clip, macro_edge_clip = get_macro_matrix(edge_clip,type='sum')
        plot_macro_matrix(macro_edge_clip,title='Macro Matrix - Zero Clipped ' + task)

        idx_max, w_max, max_mat = get_max_weights(edge,0.1)
        d_edgemax, macro_edgemax = get_macro_matrix(max_mat,type='sum')
        plot_macro_matrix(macro_edgemax,title='Macro Matrix - Max 10% weights ' + task)
        max_norm = (max_mat - max_mat.min()) / (max_mat.max() - max_mat.min())

