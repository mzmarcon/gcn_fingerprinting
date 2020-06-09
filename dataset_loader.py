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

class ACERTA_FP(Dataset):
    def __init__(self, set, split=0.8, input_type='condition', condition='None', adj_threshold=0.0):

        data_path = './'
        rst_data_file = data_path + 'rst_cn_data.hdf5'
        # task_data_file = data_path + 'task_data.hdf5'
        task_data_file = data_path + 'betas_data.hdf5'
                
        file_rst = h5py.File(rst_data_file, 'r')
        file_task = h5py.File(task_data_file, 'r')
         
        ids_rst_v1 = list(file_rst['visit1'].keys())
        ids_rst_v2 = list(file_rst['visit2'].keys())
        ids_task_v1 = list(file_task['visit1'].keys())
        ids_task_v2 = list(file_task['visit2'].keys())

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

        if set == 'training':
            if input_type == 'condition':
                self.dataset = self.process_betas_condition_dataset(file_rst,file_task,train_ids,labels_dict,adj_rst,condition)
            if input_type == 'allbetas':
                self.dataset = self.process_betas_dataset(file_rst,file_task,train_ids,labels_dict,adj_rst)


        if set == 'test':
            if input_type == 'condition':
                self.dataset = self.process_betas_condition_dataset(file_rst,file_task,test_ids,labels_dict,adj_rst,condition)
            if input_type == 'allbetas':
                self.dataset = self.process_betas_dataset(file_rst,file_task,test_ids,labels_dict,adj_rst)


    def __getitem__(self, idx):
        data_anchor = self.dataset[idx]

        #get positive example
        positive_anchor = np.random.choice(data_anchor['matching_idx'])
        data_positive=self.dataset[positive_anchor]

        #TODO np.random choices are the same every epoch because of random seed.
        #get negative example
        while True:
            n_rnd = np.random.randint(len(self.dataset))    
            if not n_rnd in [idx] + data_anchor['matching_idx']:
                data_negative = self.dataset[n_rnd]
                break
        
        return {
            'input_anchor'          : data_anchor['graph'],
            'input_positive'        : data_positive['graph'],
            'input_negative'        : data_negative['graph'],
            'matching_idx'         : data_anchor['matching_idx']
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


    def process_betas_dataset(self,file_rst,file_task,ids,labels_dict,adj_rst):
        dataset = []

        for visit in ['visit1','visit2']:
            for id in ids:
                features = torch.FloatTensor(file_task[visit][id]['betas_rois'][:])
                data = Data(x=features, edge_index=adj_rst._indices(), 
                            edge_attr=adj_rst._values(),label=torch.LongTensor(labels_dict[id]))
                data.id = (id, visit)
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

    def process_betas_condition_dataset(self,file_rst,file_task,ids,labels_dict,adj_rst,condition):
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
            for id in ids:
                features = file_task[visit][id]['betas_rois'][:]

                for n in range(range_start,range_stop):
                    feature = []
                    for item in features:
                        feature.append(item[n])
                    feature = torch.FloatTensor(feature).view(-1,1)

                    data = Data(x=feature, edge_index=adj_rst._indices(), 
                                edge_attr=adj_rst._values(),label=torch.LongTensor(labels_dict[id]))
                    data.id = (id, visit)
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

    def process_dataset_regular(self,file_rst,file_task,ids,labels_dict,adj_rst):
        dataset = []

        for visit in ['visit1','visit2']:
            for id in ids:
                features = torch.FloatTensor(file_task[visit][id]['region_values'][:]).view(-1,1)
                # data = Data(x=features, edge_index=adj_rst._indices(), 
                #             edge_attr=adj_rst._values(),label=torch.LongTensor(labels_dict[id]))

                data = { 'x' : features,
                         'label' : torch.LongTensor(labels_dict[id]),
                         'edge_index' : adj_rst._indices(),
                         'edge_attr' : adj_rst._values()
                }

                dataset.append({
                    'graph': data
                })
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
