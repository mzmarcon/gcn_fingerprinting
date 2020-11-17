import numpy as np
import torch
import h5py
from collections import defaultdict
from glob import glob
import pandas as pd

if __name__ == '__main__':

    hdf5_path = '/usr/share/datasets/acerta_data/hdf5_files/'
    data_path = '../data/'
    csv_filename = data_path+'dyslexic_labels.csv'
    psc_schools_file = data_path + 'shen_psc_task_schools.hdf5'
    psc_ambac_file = hdf5_path + 'shen_psc_task_AMBAC.hdf5'
    stimuli_path = data_path + 'stimuli2/'

    file_schools = h5py.File(psc_schools_file, 'r')
    file_ambac = h5py.File(psc_ambac_file, 'r')
    
    ids_ambac = [str(id_) for id_ in list(file_ambac.keys()[:-1])]
    ids_schools =  [str(id_) for id_ in list(file_schools.keys()[:-1])]
    sub_ids = ids_ambac + ids_schools

    labels = [0] * len(ids_ambac) + [1] * len(ids_schools)

    #save csv
    dic = {'id': sub_ids, 'label': labels}
    df = pd.DataFrame(data=dic)
    df.to_csv(csv_filename,index=False)
