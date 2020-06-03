import os
import numpy as np
import nibabel as nib
import h5py
from glob import glob
from sklearn.model_selection import train_test_split
from nilearn.input_data import NiftiLabelsMasker
from nilearn.signal import clean
from nilearn.image import high_variance_confounds
from nilearn.connectome import ConnectivityMeasure
from nilearn import plotting

def prune_timeseries(timeseries):
    #prune timeseries sizes to match smaller instance
    sizes = [] 
    for item in timeseries: 
        sizes.append(item.shape[0])
    min_size = np.min(sizes)
    for n in range(len(timeseries)):
        timeseries[n] = timeseries[n][:min_size]

    return timeseries

def get_region_timeseries(file_list, atlas, confounds=False):
    timeseries_list = []
    for file in file_list:
        masker = NiftiLabelsMasker(labels_img=atlas, standardize=True,
                                    memory='nilearn_cache', verbose=5)
        timeseries = masker.fit_transform(file)
        if confounds:
            timeseries = remove_confounds(timeseries,file)
        
        timeseries_list.append(timeseries)

    return np.array(timeseries_list)

def remove_confounds(timeseries,file):
    img = nib.load(file)
    confounds = high_variance_confounds(img,n_confounds=10)
    clean_timeseries = clean(timeseries,detrend=True,confounds=confounds)

    return clean_timeseries

def plot_correlation_matrix(correlation_matrix, atlas_data):
    np.fill_diagonal(correlation_matrix, 0) #mask the main diagonal for visualization: 
    labels = np.unique(atlas_data)
    plotting.plot_matrix(correlation_matrix, figure=(10, 8), labels=labels, 
                         vmax=1, vmin=-1, reorder=True) 
    plotting.show()

def get_correlation_matrix(region_timeseries_list):
    """
    Compute correlation matrix for a list of timeseries
    """
    correlation_matrices = []
    correlation_measure = ConnectivityMeasure(kind='correlation') 

    for item in region_timeseries_list:
        matrix = correlation_measure.fit_transform([item])[0]
        correlation_matrices.append(matrix)

    return np.array(correlation_matrices)

def create_hd5(visit1_paths,visit2_paths,cn_mx1,cn_mx2,cc200_coords):
    print("Generating hd5 file" )
    ids_visit1= []
    ids_visit2 = []
    for item in visit1_paths: 
        id = item.split('.')[-3] 
        ids_visit1.append(id)
    for item in visit2_paths:
        id = item.split('.')[-4] 
        ids_visit2.append(id) 

    hf = h5py.File('RST_cn_matrix.hdf5', 'w') 
    
    hf.create_dataset('coordinates',data=cc200_coords)
    g1 = hf.create_group('visit1') 
    g2 = hf.create_group('visit2') 
    for n in range(len(ids_visit1)): 
        subj = g1.create_group(ids_visit1[n]) 
        subj.create_dataset('cn_matrix',data=cn_mx1[n]) 
    for n in range(len(ids_visit2)): 
        subj = g2.create_group(ids_visit2[n]) 
        subj.create_dataset('cn_matrix',data=cn_mx2[n])

    hf.close()
    print("Done.")


if __name__ == '__main__':
    data_path = '/home/marcon/datasets/acerta_data/'
    visit1_paths = glob(data_path + 'acerta_RST/SCHOOLS/visit1/' + '*nii.gz')
    visit2_paths = glob(data_path + 'acerta_RST/SCHOOLS/visit2/' + '*nii.gz')
    
    mask_path = data_path + 'Masks/HaskinsPeds_NL_template_3x3x3_maskRESAMPLED.nii'
    atlas_path = data_path + 'Masks/HaskinsPeds_NL_atlasRESAMPLED1.0.nii'
    cc200_path = data_path + 'Masks/rm_group_mean_tcorr_cluster_200.nii.gz'

    cc200_coords = plotting.find_parcellation_cut_coords(labels_img=cc200_path)                      

    print("Computing timeseries")
    ts1 = get_region_timeseries(visit1_paths,cc200_path)
    ts2 = get_region_timeseries(visit2_paths,cc200_path)

    print("Computing correlation matrices")
    cn_mx1 = get_correlation_matrix(ts1)
    cn_mx2 = get_correlation_matrix(ts2) 

    create_hd5(visit1_paths,visit2_paths,cn_mx1,cn_mx2,cc200_coords)
