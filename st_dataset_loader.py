import numpy as np
import torch
import random
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torchvision import transforms
from sklearn.model_selection import train_test_split
from utils import *
import h5py
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from collections import defaultdict
from glob import glob

class ACERTA_reading_ST(Dataset):
    def __init__(self, split=0.8, condition='none', adj_threshold=0.0, window_t=320, prune=False):
        self.split = split

        data_path = 'data/'
        rst_data_file = data_path + 'rst_cn_data.hdf5'
        psc_cn_data_file = data_path + 'shen_cn_task_schools.hdf5'
        psc_data_file = data_path + 'shen_psc_task_schools.hdf5'
        reading_labels = data_path + 'reading_labels.csv'
        stimuli_path = data_path + 'stimuli2/'

        file_rst = h5py.File(rst_data_file, 'r')
        file_cn = h5py.File(psc_cn_data_file, 'r')
        file_task = h5py.File(psc_data_file, 'r')
        
        csv_file = pd.read_csv(reading_labels)
        
        # prune overrepresented dataset examples SCHM (bad readers) from csv
        sch_idx = csv_file.index[csv_file['id'].str.contains('SCHM')].tolist()
        to_remove = random.sample(sch_idx,8)
        csv_pruned = csv_file.drop(to_remove)

        # retrieve csv values
        ids = csv_pruned['id'].tolist()
        labels = csv_pruned['label'].tolist()
        visits = csv_pruned['visit'].tolist()
        
        # zip ids and visits for dataset split
        tup_list = list(zip(ids,visits))
        self.ids_visit_list = [sub_id + 'visit' + str(v) for sub_id,v in tup_list[:]]

        self.adj_matrix = self.generate_mean_adj(file_cn,ids,threshold=adj_threshold)
        # self.adj_matrix = self.generate_mean_adj_rst(file_rst,ids,threshold=adj_threshold)

        if prune:
            self.adj_matrix = prune_macro_region(self.adj_matrix,3)

        self.train_ids, self.test_ids = train_test_split(self.ids_visit_list,train_size=split,stratify=labels)

        self.dataset, self.label_dict = self.process_psc_reading_dataset(file_task,stimuli_path,ids,labels,visits,self.adj_matrix,\
                                                        window_t=window_t,condition=condition,window=True)
                
        self.train_idx=[] 
        for item in self.train_ids: 
            self.train_idx.extend(self.label_dict[item])

        self.test_idx=[] 
        for item in self.test_ids: 
            self.test_idx.extend(self.label_dict[item])
        
        if len(set(self.train_idx).intersection(self.test_idx))>0:
            raise ValueError("Dataset Double Dipping.")
        else:
            print("Dataset check.")
        print("Prune: ",prune)

    def __getitem__(self, idx):

        np.random.seed()

        data_anchor = self.dataset[idx]
        anchor_label = data_anchor['label']
        anchor_id = data_anchor['info'][0]

        coin_flip = np.random.randint(0,2)

        if coin_flip == 1: #get positive example
            label = 1 #for cross-entropy
            while True:
                rnd_positive = np.random.randint(len(self.dataset))
                if self.dataset[rnd_positive]['label'] == anchor_label: #if label is the same
                    if not self.dataset[rnd_positive]['info'][0] == anchor_id: #if not same id and visit
                        data_pair=self.dataset[rnd_positive]
                        break

        elif coin_flip == 0: #get negative example
            label = 0 
            while True:
                rnd_negative = np.random.randint(len(self.dataset))    
                if not self.dataset[rnd_negative]['label'] == anchor_label:
                    data_pair = self.dataset[rnd_negative]
                    break
        
        return {
            'input_anchor'          : data_anchor['graph'],
            'anchor_info'           : data_anchor['info'],
            'input_pair'            : data_pair['graph'],
            'pair_info'             : data_pair['info'],
            'label'                 : label,
            'label_single'          : data_anchor['label']
        }

    def __len__(self):
        return len(self.dataset)

    def process_psc_reading_dataset(self,file_task,stimuli_path,ids,labels,visits,adj_rst,window_t,condition='none',window=False):

        print("Loading PSC dataset")
        dataset = []
        element_count = 0
        #initialize label dict to save element indices for each class
        label_dict = defaultdict(list)

        base_files = sorted(glob(stimuli_path+"base*.1D"))      
        reg_files = sorted(glob(stimuli_path+"reg*.1D"))
        irr_files = sorted(glob(stimuli_path+"irr*.1D"))
        pse_files = sorted(glob(stimuli_path+"pse*.1D"))
        
        base_times = [np.loadtxt(base_files[i],dtype=int).max() for i in range(len(base_files))]
        reg_times = [np.loadtxt(reg_files[i],dtype=int).max() for i in range(len(reg_files))]
        irr_times = [np.loadtxt(irr_files[i],dtype=int).max() for i in range(len(irr_files))]
        pse_times = [np.loadtxt(pse_files[i],dtype=int).max() for i in range(len(pse_files))]

        if condition == 'irr':
            stim_times = irr_times
        elif condition == 'pse':
            stim_times = pse_times
        elif condition == 'reg':
            stim_times = reg_times
        else:
            stim_times = sorted(irr_times+reg_times+pse_times)

        for n,sub_id in enumerate(ids):
            if visits[n] == 1:
                visit = 'visit1'
            elif visits[n] == 2:
                visit = 'visit2'
                
            print("Loading Sub id {}-{}".format(sub_id,visit))
            features = file_task[sub_id][visit]['psc'][:]

            if window:
                if window_t<50:
                    if condition == 'none':
                        for onset_time in stim_times:
                            if window_t < 12:
                                onset_time = onset_time + 3
                            feature = features[onset_time:onset_time+window_t]
                            feature = torch.FloatTensor(feature)
                            dataset.append({
                                'graph': feature,
                                'label': labels[n],
                                'info': (sub_id, visit)
                            })
                            label_dict[sub_id+visit].append(element_count)
                            element_count += 1

                    elif condition == 'all':   # Generates one feature with the stims for each condition. Recommended window_t = 8
                        for stimuli in [reg_times, irr_times, pse_times]:
                            feature = []
                            for onset_time in stimuli:
                                feature.extend(features[onset_time:onset_time+window_t])
                            feature = torch.FloatTensor(feature)
                            dataset.append({
                                'graph': feature,
                                'label': labels[n],
                                'info': (sub_id, visit)
                            })
                            label_dict[sub_id+visit].append(element_count)
                            element_count += 1
                else:
                    feature = features[:window_t]
                    label_dict[sub_id+visit].append(element_count)
                    dataset.append(
                        {'graph': feature,'label': labels[n],'info': (sub_id, visit)})
                    element_count += 1

                    feature2 = features[window_t:2*window_t]
                    label_dict[sub_id+visit].append(element_count)
                    dataset.append(
                        {'graph': feature2,'label': labels[n],'info': (sub_id, visit)})
                    element_count += 1

            else:
                dataset.append({
                    'graph': features,
                    'label': labels[n],
                    'info': (sub_id, visit)
                })
                label_dict[sub_id+visit].append(element_count)
                element_count += 1

        return dataset, label_dict

    def generate_mean_adj_rst(self,file_rst,ids,threshold):
        cn_matrix_list = []

        for sub_id in ids:
            if sub_id in list(file_rst['visit1'].keys()):
                data_rst = file_rst['visit1'][sub_id]['cn_matrix'][:]
                cn_matrix_list.append(data_rst)
            if sub_id in list(file_rst['visit2'].keys()):
                data_rst = file_rst['visit2'][sub_id]['cn_matrix'][:]
                cn_matrix_list.append(data_rst)
        
        cn_matrix = np.mean(cn_matrix_list,axis=0)

        adj_rst, _ = get_adjacency(cn_matrix,threshold)

        return adj_rst

    def generate_mean_adj(self,file_cn,ids,threshold):
        cn_matrix_list = []

        for sub_id in ids:
            if 'visit1' in list(file_cn[sub_id].keys()):
                data_task = file_cn[sub_id]['visit1']['cn_matrix'][:]
                cn_matrix_list.append(data_task)
            if 'visit2' in list(file_cn[sub_id].keys()):
                data_task = file_cn[sub_id]['visit2']['cn_matrix'][:]
                cn_matrix_list.append(data_task)

        cn_matrix = np.mean(cn_matrix_list,axis=0)
        adj_rst, _ = get_adjacency(cn_matrix,threshold)

        return adj_rst

"""""""""""""""
Dyslexia
"""""""""""""""

class ACERTA_dyslexic_ST(Dataset):
    """
    Dataset class for dyslexia classification
    """
    def __init__(self, split=0.8, condition='all', adj_threshold=0.0, window_t=300, prune=False):
        self.split = split

        data_path = 'data/'
        hdf5_path = '/usr/share/datasets/acerta_data/hdf5_files/'
        rst_data_file = data_path + 'rst_cn_data.hdf5'
        schools_cn_data_file = data_path + 'shen_cn_task_schools.hdf5'
        psc_schools_file = data_path + 'shen_psc_task_schools.hdf5'
        ambac_cn_data_file = data_path + 'shen_cn_task_AMBAC.hdf5'
        psc_ambac_file = hdf5_path + 'shen_psc_task_AMBAC.hdf5'
        dyslexic_labels = data_path + 'dyslexic_labels.csv'
        stimuli_path = data_path + 'stimuli2/'

        file_rst = h5py.File(rst_data_file, 'r')
        file_cn_schools = h5py.File(schools_cn_data_file, 'r')        
        file_cn_ambac = h5py.File(ambac_cn_data_file, 'r')        
        file_schools = h5py.File(psc_schools_file, 'r')
        file_ambac = h5py.File(psc_ambac_file, 'r')
        
        csv_file = pd.read_csv(dyslexic_labels)
        self.ids = csv_file['id'].tolist()
        labels = csv_file['label'].tolist()

        # self.adj_matrix = self.generate_mean_adj_rst(file_rst,self.ids,threshold=adj_threshold)
        self.adj_matrix = self.generate_mean_adj(file_cn_schools,file_cn_ambac,self.ids,threshold=adj_threshold)
        
        if prune:
            self.adj_matrix = prune_macro_region(self.adj_matrix,3)
        
        train_ids, test_ids = train_test_split(self.ids,train_size=split,stratify=labels)

        self.dataset, label_dict = self.process_psc_reading_dataset(file_schools,file_ambac,stimuli_path,self.ids,labels,self.adj_matrix,\
                                                        window_t=window_t,condition=condition,window=True)

        self.train_idx=[] 
        for item in train_ids: 
            self.train_idx.extend(label_dict[item])

        self.test_idx=[] 
        for item in test_ids: 
            self.test_idx.extend(label_dict[item])

        if len(set(self.train_idx).intersection(self.test_idx))>0:
            raise ValueError("Dataset Double Dipping.")
        else:
            print("Dataset check.")
        print("Prune: ",prune)

    def __getitem__(self, idx):

        np.random.seed()

        data_anchor = self.dataset[idx]
        anchor_label = data_anchor['label']
        anchor_id = data_anchor['info'][0]

        coin_flip = np.random.randint(0,2)

        if coin_flip == 1: #get positive example
            label = 1 #for cross-entropy
            while True:
                rnd_positive = np.random.randint(len(self.dataset))
                if self.dataset[rnd_positive]['label'] == anchor_label: #if label is the same
                    if not self.dataset[rnd_positive]['info'][0] == anchor_id: #if not same id and visit
                        data_pair=self.dataset[rnd_positive]
                        break

        elif coin_flip == 0: #get negative example
            label = 0 
            while True:
                rnd_negative = np.random.randint(len(self.dataset))    
                if not self.dataset[rnd_negative]['label'] == anchor_label:
                    data_pair = self.dataset[rnd_negative]
                    break
        
        return {
            'input_anchor'          : data_anchor['graph'],
            'anchor_info'           : data_anchor['info'],
            'input_pair'            : data_pair['graph'],
            'pair_info'             : data_pair['info'],
            'label'                 : label,
            'label_single'          : data_anchor['label']
        }

    def __len__(self):
        return len(self.dataset)

    def process_psc_reading_dataset(self,file_schools,file_ambac,stimuli_path,ids,labels,adj_rst,window_t=12,condition='all',window=False):

        print("Loading PSC dataset")
        dataset = []
        element_count = 0

        #initialize label dict to save element indices for each class
        label_dict = defaultdict(list)

        base_files = sorted(glob(stimuli_path+"base*.1D"))      
        reg_files = sorted(glob(stimuli_path+"reg*.1D"))
        irr_files = sorted(glob(stimuli_path+"irr*.1D"))
        pse_files = sorted(glob(stimuli_path+"pse*.1D"))
        
        base_times = [np.loadtxt(base_files[i],dtype=int).max() for i in range(len(base_files))]
        reg_times = [np.loadtxt(reg_files[i],dtype=int).max() for i in range(len(reg_files))]
        irr_times = [np.loadtxt(irr_files[i],dtype=int).max() for i in range(len(irr_files))]
        pse_times = [np.loadtxt(pse_files[i],dtype=int).max() for i in range(len(pse_files))]

        if condition == 'irr':
            stim_times = irr_times
        elif condition == 'pse':
            stim_times = pse_times
        elif condition == 'reg':
            stim_times = reg_times
        else:
            stim_times = sorted(irr_times+reg_times+pse_times)

        visit = 'visit1'
        for n,sub_id in enumerate(ids):        
            print("Loading Sub id {}-{}".format(sub_id,visit))
            if 'AMBAC' in sub_id:
                features = file_ambac[sub_id][visit]['psc'][:]
            elif 'SCH' in sub_id:
                features = file_schools[sub_id][visit]['psc'][:]
            else:
                raise ValueError("Wrong subject identifier.")

            if window:
                if window_t<50:
                    for onset_time in stim_times:
                        feature = []
                        for timestamp in range(onset_time,onset_time+window_t):
                            feature.append(features[timestamp])
                        feature = torch.FloatTensor(feature)

                        dataset.append({
                            'graph': feature,
                            'label': labels[n],
                            'info': (sub_id, visit)
                        })
                else:
                    feature = features[:window_t]
                    label_dict[sub_id].append(element_count)
                    dataset.append(
                        {'graph': feature,'label': labels[n],'info': (sub_id, visit)})
                    element_count += 1

                    feature2 = features[window_t:2*window_t]
                    label_dict[sub_id].append(element_count)
                    dataset.append(
                        {'graph': feature2,'label': labels[n],'info': (sub_id, visit)})
                    element_count += 1
            else:
                dataset.append({
                    'graph': features,
                    'label': labels[n],
                    'info': (sub_id, visit)
                })

        return dataset,label_dict

    def generate_mean_adj_rst(self,file_rst,ids,threshold):
        cn_matrix_list = []

        for sub_id in ids:
            if sub_id in list(file_rst['visit1'].keys()):
                data_rst = file_rst['visit1'][sub_id]['cn_matrix'][:]
                cn_matrix_list.append(data_rst)
            if sub_id in list(file_rst['visit2'].keys()):
                data_rst = file_rst['visit2'][sub_id]['cn_matrix'][:]
                cn_matrix_list.append(data_rst)
        
        cn_matrix = np.mean(cn_matrix_list,axis=0)

        adj_rst, _ = get_adjacency(cn_matrix,threshold)

        return adj_rst

    def generate_mean_adj(self,file_cn_schools,file_cn_ambac,ids,threshold):
        cn_matrix_list = []

        for sub_id in ids:
            if sub_id in list(file_cn_schools.keys()):
                data_rst = file_cn_schools[sub_id]['visit1']['cn_matrix'][:]
                cn_matrix_list.append(data_rst)

            if sub_id in list(file_cn_ambac.keys()):
                data_rst = file_cn_ambac[sub_id]['visit1']['cn_matrix'][:]
                cn_matrix_list.append(data_rst)

        cn_matrix = np.mean(cn_matrix_list,axis=0)
        adj_matrix, _ = get_adjacency(cn_matrix,threshold)

        return adj_matrix
