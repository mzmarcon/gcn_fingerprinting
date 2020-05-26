import numpy as np
import h5py
from glob import glob
import nibabel as nib
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
        #create list of mean values for each atlas region
        subject_means = []
        for index in indices[1:]:
            index_values = []
            for item in index_dict[index]:
                #get voxel values and mean for each index
                index_values.append(fdata_flatten[item])
                index_mean = np.mean(index_values)
            subject_means.append(index_mean)
        region_means.append(subject_means)

    return np.swapaxes(region_means,0,1)

def append_hd5(folder_list):
    for subject_folder in v2_folders:
        sub_id = subject_folder.split('/')[-1]
        print("Processing subject: ",sub_id)
        files = sorted(glob(subject_folder + '/*.nii.gz')) #sort to irr0..19, pse0..19, reg0..19
        betas_per_roi = extract_region_values(files,cc200_path) #get mean beta value per roi for each stimulus [200,60]
        subj = hf.create_group('/visit2/'+sub_id) 
        subj.create_dataset('betas_rois',data=betas_per_roi) 

if __name__ == '__main__':
    v1_path = '/usr/share/datasets/SCHOOLS/SCHBNOVO/BETAS_7s/'
    v2_path = '/usr/share/datasets/acerta_data/acerta_TASK/SCHOOLS_Betas/visit2/'
    v1_folders = glob(v1_path + 'betas*')
    v2_folders = glob(v2_path + '*')

    cc200_path = '/home/marcon/datasets/acerta_data/Masks/rm_group_mean_tcorr_cluster_200.nii.gz'
    cc200_coords = plotting.find_parcellation_cut_coords(labels_img=cc200_path)

    hf = h5py.File('betas_new_data.hdf5', 'w') 
    hf.create_dataset('coordinates',data=cc200_coords)
    g1 = hf.create_group('visit1') 
    g2 = hf.create_group('visit2') 

    #store visit1 values
    for subject_folder in v1_folders:
        sub_id = subject_folder.split('_')[-1]
        print("Processing subject: ",sub_id)
        files = sorted(glob(subject_folder + '/*.nii')) #sort to irr0..19, pse0..19, reg0..19
        betas_per_roi = extract_region_values(files,cc200_path) #get mean beta value per roi for each stimulus [200,60]
        subj = g1.create_group(sub_id) 
        subj.create_dataset('betas_rois',data=betas_per_roi) 

    #store visit2 values    
    for subject_folder in v2_folders:
        sub_id = subject_folder.split('/')[-1]
        print("Processing subject: ",sub_id)
        files = sorted(glob(subject_folder + '/*.nii.gz')) #sort to irr0..19, pse0..19, reg0..19
        betas_per_roi = extract_region_values(files,cc200_path) #get mean beta value per roi for each stimulus [200,60]
        subj = g2.create_group(sub_id) 
        subj.create_dataset('betas_rois',data=betas_per_roi) 