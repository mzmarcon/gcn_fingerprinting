import numpy as np
import torch
import h5py
from collections import defaultdict
from glob import glob
import pandas as pd

if __name__ == '__main__':

    data_path = '../data/'
    csv_filename = data_path+'reading_labels.csv'
    rst_data_file = data_path + 'rst_cn_data.hdf5'
    psc_data_file = data_path + 'shen_psc_task_schools.hdf5'
    stimuli_path = data_path + 'stimuli2/'

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

    good_rst_v1 = [id_ for id_ in ids_rst_v1 if 'B' in id_] 
    good_rst_v2 = [id_ for id_ in ids_rst_v2 if 'B' in id_] 
    good_task_v1 = [id_ for id_ in ids_task_v1 if 'B' in id_] 
    good_task_v2 = [id_ for id_ in ids_task_v2 if 'B' in id_] 

    bad_rst_v1 = [id_ for id_ in ids_rst_v1 if 'M' in id_] 
    bad_rst_v2 = [id_ for id_ in ids_rst_v2 if 'M' in id_] 
    bad_task_v1 = [id_ for id_ in ids_task_v1 if 'M' in id_] 
    bad_task_v2 = [id_ for id_ in ids_task_v2 if 'M' in id_] 

    common_good_v1 = list(set(good_task_v1).intersection(good_rst_v1))
    common_good_v2 = list(set(good_task_v2).intersection(good_rst_v2))
    common_bad_v1 = list(set(bad_task_v1).intersection(bad_rst_v1))
    common_bad_v2 = list(set(bad_task_v2).intersection(bad_rst_v2))

    #make labels and visit labels for stratification
    labels = [0]*(len(common_good_v1)+len(common_good_v2)) + [1]*(len(common_bad_v1)+len(common_bad_v2))
    visit_labels = [1]*len(common_good_v1)+[2]*len(common_good_v2)+[1]*len(common_bad_v1)+[2]*len(common_bad_v2)

    good_ids = common_good_v1 + common_good_v2
    bad_ids = common_bad_v1 + common_bad_v2
    sub_ids = good_ids + bad_ids

    id_numbers = list(range(len(sub_ids)))

    sub_list = []
    for n in id_numbers:
        sub_list.append((sub_ids[n],visit_labels[n]))

    ids_v1 = common_good_v1 + common_bad_v1
    ids_v2 = common_good_v2 + common_bad_v2

    #save csv
    dic = {'id': sub_ids, 'label': labels, 'visit': visit_labels}
    df = pd.DataFrame(data=dic)
    df.to_csv(csv_filename,index=False)
