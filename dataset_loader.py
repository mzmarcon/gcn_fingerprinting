import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torchvision import transforms
from sklearn.model_selection import train_test_split
from utils import *
import h5py
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
from collections import defaultdict
from glob import glob


class ACERTA_FP(Dataset):
    def __init__(self, set_split, split=0.8, input_type='betas', condition='None', adj_threshold=0.0):
        self.set = set_split

        data_path = 'data/'
        rst_data_file = data_path + 'rst_cn_data.hdf5'
        betas_data_file = data_path + 'betas_data.hdf5'
        psc_data_file = data_path + 'shen_psc_task_schools.hdf5'
        stimuli_path = data_path + 'stimuli2/'
                

        if input_type == 'betas':
            file_rst = h5py.File(rst_data_file, 'r')
            file_task = h5py.File(betas_data_file, 'r')
         
            ids_rst_v1 = list(file_rst['visit1'].keys())
            ids_rst_v2 = list(file_rst['visit2'].keys())
            ids_task_v1 = list(file_task['visit1'].keys())
            ids_task_v2 = list(file_task['visit2'].keys())

        elif input_type == 'PSC':
            file_rst = h5py.File(rst_data_file, 'r')
            file_task = h5py.File(psc_data_file, 'r')
         
            ids_rst_v1 = list(file_rst['visit1'].keys())
            ids_rst_v2 = list(file_rst['visit2'].keys())

            task_subjects = list(file_task.keys())
            ids_task_v1 = []
            ids_task_v2 = []
            for subject in task_subjects:
                if 'visit1' in list(file_task[subject].keys()):
                    ids_task_v1.append(subject)
                if 'visit2' in list(file_task[subject].keys()):
                    ids_task_v2.append(subject)

        common_v1 = self.common_elements(ids_rst_v1,ids_task_v1) 
        common_v2 = self.common_elements(ids_rst_v2,ids_task_v2)
        common_ids = self.common_elements(common_v1,common_v2)

        adj_rst = self.generate_mean_adj(file_rst,common_ids,threshold=adj_threshold)

        enc = OneHotEncoder(handle_unknown='ignore')
        enc_ids = enc.fit_transform(np.array(common_ids).reshape(-1,1)).toarray()

        labels_dict = defaultdict(list)
        for n in range(len(common_ids)):
            labels_dict[common_ids[n]] = enc_ids[n] 

        train_ids, test_ids = self.split_train_test(common_ids,size=split)

        if self.set == 'training':
            if input_type == 'betas':
                self.dataset = self.process_betas_condition_dataset(file_task,train_ids,labels_dict,adj_rst,condition)
            if input_type == 'PSC':
                self.dataset = self.process_psc_dataset(file_task,stimuli_path,train_ids,labels_dict,adj_rst,condition=condition)


        if self.set == 'test':
            if input_type == 'betas':
                self.dataset = self.process_betas_condition_dataset(file_task,test_ids,labels_dict,adj_rst,condition)
            if input_type == 'PSC':
                self.dataset = self.process_psc_dataset(file_task,stimuli_path,test_ids,labels_dict,adj_rst,condition=condition)


    def __getitem__(self, idx):

        np.random.seed()

        data_anchor = self.dataset[idx]
        positive_anchor = np.random.choice(data_anchor['matching_idx'])
        data_positive=self.dataset[positive_anchor]

        while True:
            n_rnd = np.random.randint(len(self.dataset))    
            if not n_rnd in [idx] + data_anchor['matching_idx']:
                data_negative = self.dataset[n_rnd]
                break

        coin_flip = np.random.randint(0,2)        
        if coin_flip == 1: #get positive example   
            label = 0 #for contrastive
            data_pair = data_positive

        elif coin_flip == 0: #get negative example
            label = 1
            data_pair = data_negative

        return {
            'input_anchor'          : data_anchor['graph'],
            'input_positive'        : data_positive['graph'],
            'input_negative'        : data_negative['graph'],
            'input_pair'            : data_pair['graph'],
            'label'                 : label,
            'matching_idx'          : data_anchor['matching_idx']
        }


    def __len__(self):
        return len(self.dataset)


    def common_elements(self,list1,list2):
        common = [] 
        for item in list1: 
            if item in list2: 
                common.append(item)  
        return common


    def split_train_test(self,id_list,size=0.8,random_seed=42):
        np.random.seed(random_seed)
        n_split = int(size*len(id_list))
        train = list(np.random.choice(id_list,n_split,replace=False))
        test = []
        for item in id_list:
            if item not in train:
                test.append(item)

        return train, test


    def process_betas_dataset(self,file_task,ids,labels_dict,adj_rst):
        dataset = []

        for visit in ['visit1','visit2']:
            for sub_id in ids:
                features = torch.FloatTensor(file_task[visit][sub_id]['betas_rois'][:])
                data = Data(x=features, edge_index=adj_rst._indices(), 
                            edge_attr=adj_rst._values(),label=torch.LongTensor(labels_dict[sub_id]))
                data.id = (sub_id, visit)
                dataset.append({
                    'graph': data
                })
        
        for n_anchor in range(len(dataset)): #get matching example indices for each example
            matching_ids = []
            for n_pair in range(len(dataset)):
                if not n_anchor == n_pair:  #remove self loops
                    if dataset[n_anchor]['graph']['id'][0] ==  dataset[n_pair]['graph']['id'][0]: #check if same id
                        if not dataset[n_anchor]['graph']['id'][1] ==  dataset[n_pair]['graph']['id'][1]: #force visit1to2
                            matching_ids.append(n_pair)

            dataset[n_anchor]['matching_idx'] = matching_ids
        
        return dataset

    def process_betas_condition_dataset(self,file_task,ids,labels_dict,adj_rst,condition):
        dataset = []

        if condition == 'irr':
            range_start = 0
            range_stop = 20
        elif condition == 'pse':
            range_start = 20
            range_stop = 40
        elif condition == 'reg':
            range_start = 40
            range_stop = 60
        else:
            range_start=0
            range_stop=60

        for visit in ['visit1','visit2']:
            for sub_id in ids:
                features = file_task[visit][sub_id]['betas_rois'][:]

                for n in range(range_start,range_stop):
                    feature = []
                    for item in features:
                        feature.append(item[n])
                    feature = torch.FloatTensor(feature).view(-1,1)

                    data = Data(x=feature, edge_index=adj_rst._indices(), 
                                edge_attr=adj_rst._values(),label=torch.LongTensor(labels_dict[sub_id]))
                    data.id = (sub_id, visit)
                    dataset.append({
                        'graph': data
                    })

        for n_anchor in range(len(dataset)): #get matching example indices for each example
            matching_ids = []
            for n_pair in range(len(dataset)):
                if not n_anchor == n_pair:  #remove self loops
                    if dataset[n_anchor]['graph']['id'][0] ==  dataset[n_pair]['graph']['id'][0]: #check same id
                        if not dataset[n_anchor]['graph']['id'][1] ==  dataset[n_pair]['graph']['id'][1]: #force visit1to2
                            matching_ids.append(n_pair)

            dataset[n_anchor]['matching_idx'] = matching_ids

        return dataset


    def process_psc_dataset(self,file_task,stimuli_path,ids,labels_dict,adj_rst,window_t=7,condition='all',remove_baseline=False):

        print("Loading PSC dataset")
        dataset = []

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

        for subject_id in ids:
            print("Loading subject: ",subject_id)
            for visit in ['visit1','visit2']:
                features = file_task[subject_id][visit]['psc'][:]

                for onset_time in stim_times:
                    feature = []

                    for timestamp in range(onset_time,onset_time+window_t):
                        feature.append(features[timestamp])
                    feature = np.swapaxes(feature,0,1)
                    feature = torch.FloatTensor(feature)

                    #TODO remove baseline

                    data = Data(x=feature, edge_index=adj_rst._indices(), 
                                edge_attr=adj_rst._values(),label=torch.LongTensor(labels_dict[subject_id]))
                    data.id = (subject_id, visit)
                    data.time = onset_time
                    dataset.append({
                        'graph': data
                    })

        for n_anchor in range(len(dataset)): #get matching example indices for each example
            matching_ids = []
            for n_pair in range(len(dataset)):
                if not n_anchor == n_pair:  #remove self loops
                    if dataset[n_anchor]['graph']['id'][0] ==  dataset[n_pair]['graph']['id'][0]: #check same id
                        if not dataset[n_anchor]['graph']['id'][1] ==  dataset[n_pair]['graph']['id'][1]: #force visit1to2
                            matching_ids.append(n_pair)

            dataset[n_anchor]['matching_idx'] = matching_ids

        return dataset

    def generate_mean_adj(self,file_rst,ids,threshold):
        cn_matrix_list = []
        for visit in ['visit1','visit2']:
            for id in ids:
                data_rst = file_rst[visit][id]['cn_matrix'][:]
                cn_matrix_list.append(data_rst)
        
        cn_matrix = np.mean(cn_matrix_list,axis=0)

        mask_rst, _ = get_adjacency(cn_matrix,threshold)
        adj_rst = sparse.coo_matrix(mask_rst)
        adj_rst = torch.sparse_coo_tensor(torch.LongTensor(np.vstack((adj_rst.row, adj_rst.col))),
                                            torch.FloatTensor(adj_rst.data),
                                            adj_rst.shape)

        return adj_rst



#------------ Reading score classification ------------


class ACERTA_reading(Dataset):
    def __init__(self, set_split, split=0.8, input_type='betas', condition='None', adj_threshold=0.0):
        self.split = split
        self.set = set_split

        data_path = 'data/'
        rst_data_file = data_path + 'rst_cn_data.hdf5'
        betas_data_file = data_path + 'betas_data.hdf5'
        psc_data_file = data_path + 'shen_psc_task_schools.hdf5'
        stimuli_path = data_path + 'stimuli2/'

        if input_type == 'betas':
            file_rst = h5py.File(rst_data_file, 'r')
            file_task = h5py.File(betas_data_file, 'r')
         
            ids_rst_v1 = list(file_rst['visit1'].keys())
            ids_rst_v2 = list(file_rst['visit2'].keys())
            ids_task_v1 = list(file_task['visit1'].keys())
            ids_task_v2 = list(file_task['visit2'].keys())

        elif input_type == 'PSC':
            file_rst = h5py.File(rst_data_file, 'r')
            file_task = h5py.File(psc_data_file, 'r')
         
            ids_rst_v1 = list(file_rst['visit1'].keys())
            ids_rst_v2 = list(file_rst['visit2'].keys())

            task_subjects = list(file_task.keys())
            ids_task_v1 = []
            ids_task_v2 = []
            for subject in task_subjects:
                if 'visit1' in list(file_task[subject].keys()):
                    ids_task_v1.append(subject)
                if 'visit2' in list(file_task[subject].keys()):
                    ids_task_v2.append(subject)

        sub_list, ids_v1, ids_v2, train_ids, \
        test_ids, train_labels, test_labels = self.get_labels_READING(ids_rst_v1,ids_rst_v2,ids_task_v1,ids_task_v2)

        adj_rst = self.generate_mean_adj(file_rst,ids_v1,ids_v2,threshold=adj_threshold)

        if self.set == 'training':
            if input_type == 'betas':
                self.dataset = self.process_betas_reading_dataset(file_task,sub_list,train_ids,train_labels,adj_rst,condition)
            elif input_type == 'PSC':
                self.dataset = self.process_psc_reading_dataset(file_task,stimuli_path,sub_list,train_ids,train_labels,adj_rst,condition=condition)
                
        if self.set == 'test':
            if input_type == 'betas':
                self.dataset = self.process_betas_reading_dataset(file_task,sub_list,test_ids,test_labels,adj_rst,condition)
            elif input_type == 'PSC':
                self.dataset = self.process_psc_reading_dataset(file_task,stimuli_path,sub_list,test_ids,test_labels,adj_rst,condition=condition)

    def __getitem__(self, idx):

        np.random.seed()

        data_anchor = self.dataset[idx]
        anchor_label = data_anchor['graph']['label']
        anchor_id = data_anchor['graph']['id']

        coin_flip = np.random.randint(0,2)

        if coin_flip == 1: #get positive example
            label = 1 #for cross-entropy.
            while True:
                rnd_positive = np.random.randint(len(self.dataset))
                if self.dataset[rnd_positive]['graph']['label'] == anchor_label: #if label is the same
                    if not self.dataset[rnd_positive]['graph']['id'] == anchor_id: #if not same id
                        data_pair=self.dataset[rnd_positive]
                        break

        elif coin_flip == 0: #get negative example
            label = 0
            while True:
                rnd_negative = np.random.randint(len(self.dataset))    
                if not self.dataset[rnd_negative]['graph']['label'] == anchor_label:
                    data_pair = self.dataset[rnd_negative]
                    break
        
        return {
            'input_anchor'          : data_anchor['graph'],
            'input_pair'            : data_pair['graph'],
            'label'                 : label,
            'label_single'          : data_anchor['graph']['label'],
            'anchor_id'             : data_anchor['graph']['id'],
            'pair_id'               : data_pair['graph']['id']
        }

    def __len__(self):
        return len(self.dataset)


    def common_elements(self,list1,list2):
        common = [] 
        for item in list1: 
            if item in list2: 
                common.append(item)  
        return common


    def process_betas_reading_dataset(self,file_task,sub_list,ids,labels,adj_rst,condition):
        dataset = []
     
        if condition == 'irr':
            range_start = 0
            range_stop = 20
        elif condition == 'pse':
            range_start = 20
            range_stop = 40
        elif condition == 'reg':
            range_start = 40
            range_stop = 60
        else:
            range_start=0
            range_stop=60

        for n in range(len(ids)):
            sub_id = sub_list[ids[n]][0]
            if sub_list[ids[n]][1] == 1:
                visit = 'visit1'
            elif sub_list[ids[n]][1] == 2:
                visit = 'visit2'
            
            # print("Sub id {}\nVisit {}".format(sub_id,visit))
            
            features = file_task[visit][sub_id]['betas_rois'][:]

            for beta_stim in range(range_start,range_stop):
                feature = []
                for item in features:
                    feature.append(item[beta_stim])
                feature = torch.FloatTensor(feature).view(-1,1)

                data = Data(x=feature, edge_index=adj_rst._indices(), 
                            edge_attr=adj_rst._values(),label=labels[n])
                data.id = (sub_id, visit)
                dataset.append({
                    'graph': data
                })

        return dataset

    def process_psc_reading_dataset(self,file_task,stimuli_path,sub_list,ids,labels,adj_rst,window_t=7,condition='all',remove_baseline=False):

        print("Loading PSC {} dataset".format(self.set))
        dataset = []

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

        for n in range(len(ids)):
            sub_id = sub_list[ids[n]][0]
            if sub_list[ids[n]][1] == 1:
                visit = 'visit1'
            elif sub_list[ids[n]][1] == 2:
                visit = 'visit2'
                
            print("Loading subject: ",sub_id)
            for visit in ['visit1','visit2']:
                features = file_task[sub_id][visit]['psc'][:]

                for onset_time in stim_times:
                    feature = []

                    for timestamp in range(onset_time,onset_time+window_t):
                        feature.append(features[timestamp])
                    feature = np.swapaxes(feature,0,1)
                    feature = torch.FloatTensor(feature)

                    #TODO remove baseline

                    data = Data(x=feature, edge_index=adj_rst._indices(), 
                                edge_attr=adj_rst._values(),label=labels[n])
                    data.id = (sub_id, visit)
                    data.time = onset_time
                    dataset.append({
                        'graph': data
                    })

        return dataset

    def get_labels_READING(self, ids_rst_v1,ids_rst_v2,ids_task_v1,ids_task_v2):
        bom_rst_v1 = self.get_reading_classes(ids_rst_v1,class_type='B') 
        bom_rst_v2 = self.get_reading_classes(ids_rst_v2,class_type='B') 
        bom_task_v1 = self.get_reading_classes(ids_task_v1,class_type='B') 
        bom_task_v2 = self.get_reading_classes(ids_task_v2,class_type='B') 

        mau_rst_v1 = self.get_reading_classes(ids_rst_v1,class_type='M') 
        mau_rst_v2 = self.get_reading_classes(ids_rst_v2,class_type='M') 
        mau_task_v1 = self.get_reading_classes(ids_task_v1,class_type='M') 
        mau_task_v2 = self.get_reading_classes(ids_task_v2,class_type='M') 

        common_bom_v1 = self.common_elements(bom_task_v1,bom_rst_v1)
        common_bom_v2 = self.common_elements(bom_task_v2,bom_rst_v2)
        common_mau_v1 = self.common_elements(mau_task_v1,mau_rst_v1)
        common_mau_v2 = self.common_elements(mau_task_v2,mau_rst_v2)

        #make labels and visit labels for stratification
        labels = [0]*(len(common_bom_v1)+len(common_bom_v2)) + [1]*(len(common_mau_v1)+len(common_mau_v2))
        visit_labels = [1]*len(common_bom_v1)+[2]*len(common_bom_v2)+[1]*len(common_mau_v1)+[2]*len(common_mau_v2)

        bom_ids = common_bom_v1 + common_bom_v2
        mau_ids = common_mau_v1 + common_mau_v2
        sub_ids = bom_ids + mau_ids

        id_numbers = list(range(len(sub_ids)))

        train_ids, test_ids, train_labels, test_labels = train_test_split(id_numbers, labels, train_size=self.split, random_state=42)

        sub_list = []
        for n in id_numbers:
            sub_list.append((sub_ids[n],visit_labels[n]))

        #generate adjaency
        ids_v1 = common_bom_v1 + common_mau_v1
        ids_v2 = common_bom_v2 + common_mau_v2

        return sub_list, ids_v1, ids_v2, train_ids, test_ids, train_labels, test_labels


    def get_reading_classes(self, ids_list, class_type='B'):
        class_list = []
        for item in ids_list:
            if class_type in item:
                class_list.append(item)
                
        return class_list

    def generate_mean_adj(self,file_rst,ids_v1,ids_v2,threshold):
        cn_matrix_list = []

        for sub_id in ids_v1:
            visit='visit1'
            data_rst = file_rst[visit][sub_id]['cn_matrix'][:]
            cn_matrix_list.append(data_rst)

        for sub_id in ids_v2:
            visit='visit2'
            data_rst = file_rst[visit][sub_id]['cn_matrix'][:]
            cn_matrix_list.append(data_rst)
        
        cn_matrix = np.mean(cn_matrix_list,axis=0)

        mask_rst, _ = get_adjacency(cn_matrix,threshold)
        adj_rst = sparse.coo_matrix(mask_rst)
        adj_rst = torch.sparse_coo_tensor(torch.LongTensor(np.vstack((adj_rst.row, adj_rst.col))),
                                            torch.FloatTensor(adj_rst.data),
                                            adj_rst.shape)

        return adj_rst
