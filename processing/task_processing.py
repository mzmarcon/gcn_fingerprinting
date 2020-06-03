import os
import numpy as np
import nibabel as nib
import h5py
from glob import glob
from nilearn import plotting
from collections import defaultdict

def extract_region_values(file_list, atlas_path):
    atlas_data = nib.load(atlas_path).get_fdata()
    indices = np.unique(atlas_data)
    altas_flat = atlas_data.flatten()

    #get indices for the voxels of each label
    print("Calculating atlas indices")
    index_dict = defaultdict(list)
    for index in indices[1:]:
        for n in range(len(altas_flat)):
            if altas_flat[n]==index:
                index_dict[index].append(n)
    #get mean value for each region for each subject
    region_means = []
    for file in file_list:
        print("Calculating mean values for subject: {}".format(file.split('.')[-3]))
        fdata = nib.load(file).get_fdata()
        fdata_flatten = fdata.flatten()
        subject_means = []
        for index in indices[1:]:
            index_values = []
            for item in index_dict[index]:
                #get voxel values and mean for each index
                index_values.append(fdata_flatten[item])
                index_mean = np.mean(index_values)
            subject_means.append(index_mean)
        region_means.append(subject_means)

    return region_means

def create_hd5(visit1_paths,visit2_paths,data_v1,data_v2,coordinates):
    print("Generating hd5 file" )
    ids_visit1= []
    ids_visit2 = []
    for item in visit1_paths: 
        id = item.split('.')[-3].split('_')[0]
        ids_visit1.append(id)
    for item in visit2_paths:
        id = item.split('.')[-3].split('_')[0]
        ids_visit2.append(id) 

    hf = h5py.File('task_data.hdf5', 'w') 
    
    hf.create_dataset('coordinates',data=coordinates)
    g1 = hf.create_group('visit1') 
    g2 = hf.create_group('visit2') 
    for n in range(len(ids_visit1)): 
        subj = g1.create_group(ids_visit1[n]) 
        subj.create_dataset('region_values',data=data_v1[n]) 
    for n in range(len(ids_visit2)): 
        subj = g2.create_group(ids_visit2[n]) 
        subj.create_dataset('region_values',data=data_v2[n])

    hf.close()
    print("Done.")

if __name__ == '__main__':
    task_path = '/usr/share/datasets/acerta_data/acerta_TASK/SCHOOLS/'
    visit1_paths = glob(task_path + 'visit1/Full_Fstat/*nii.gz')
    visit2_paths = glob(task_path + 'visit2/Full_Fstat/*nii.gz')
    
    cc200_path = '/home/marcon/datasets/acerta_data/Masks/rm_group_mean_tcorr_cluster_200.nii.gz'
    cc200_coords = plotting.find_parcellation_cut_coords(labels_img=cc200_path)

    regions_v1 = extract_region_values(visit1_paths,cc200_path)
    regions_v2 = extract_region_values(visit2_paths,cc200_path)

    create_hd5(visit1_paths,visit2_paths,regions_v1,regions_v2,cc200_coords)