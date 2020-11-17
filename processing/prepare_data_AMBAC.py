import argparse
import numpy as np
import pandas as pd
import glob
import re
import h5py
import os
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure


parser = argparse.ArgumentParser()
parser.add_argument('--visits', default='visit1', #action='append',
                    help='visits for matrix calculation')
parser.add_argument('--base_dir', type=str, default='/usr/share/datasets/acerta_data/',
                    help='path and filename for the parcellation to be used')                   
parser.add_argument('--in_rois', type=str, default='Masks/shen_1mm_268_parcellation_LPI.nii.gz',
                    help='path and filename for the parcellation to be used')
parser.add_argument('--in_files', type=str, default='acerta_TASK/AMBAC/PROC.PALA.MNI.PSC/',
                    help='path and filename for the parcellation to be used')
parser.add_argument('--subjects_ids', type=str, default='../subjects_ids.csv',
                    help='path and filename for the subjects ids file to be used')
parser.add_argument('--output_file', type=str, default='hdf5_files/shen_psc_task_AMBAC.hdf5',
                    help='file to store the connectivity matrices output')
parser.add_argument('--nifti_type', type=str, default='psc',
                    help='data on the nifti file', choices=['errts','psc'])
parser.add_argument('--stims_folder', type=str, default='stimuli/',
                    help='folder containing the stim times')

#-----------------------------------------------------------------------------------------------
args = parser.parse_args()
visits = ['visit1'] #args.nifti_folder
base_dir = args.base_dir
in_rois = base_dir + args.in_rois
in_files = base_dir + args.in_files
output_file = base_dir + args.output_file
stims_folder = base_dir + args.stims_folder
nifti_type = args.nifti_type

if not isinstance(visits,list):
    visits = [visits]

# Instance output file
output_file = h5py.File(output_file,'a')

# Load clinical data
subject_ids = [id_.split('.')[1] for id_ in os.listdir(in_files+visits[0])]

# Create roi and correlation object
masker = NiftiLabelsMasker(labels_img=in_rois, standardize=True)
correlation_measure = ConnectivityMeasure(kind='correlation')

for visit in visits:
    if nifti_type == 'errts':
        nifti_files = glob.glob(in_files+visit+'/*+tlrc.BRIK')
        print(visit)
        for subj_id in subject_ids:
            subject = subj_id[0]
            subject_file = list(filter(re.compile(".*"+subject).match,nifti_files))
            if subject_file:
                print("Preparing subject {}".format(subject))
                print("Loading time series...")
                time_series = masker.fit_transform(subject_file[0])
                print("Computing correlations...")
                correlation_matrix = correlation_measure.fit_transform([time_series])[0]
                print("Saving data...")
                try:
                    subject_group = output_file.require_group(name=subject)
                    visit_group = subject_group.create_group(name=visit)
                    visit_group.create_dataset(name='cn_matrix',data=correlation_matrix)
                except ValueError:
                    print("Duplicated subject!")
                print("Subject {} done!...".format(subject))
    if nifti_type == 'psc':
        nifti_files = glob.glob(in_files+visit+'/PSC.*')
        stim_times = output_file.require_group(name='stim_times')
        stim_times.create_dataset(name='base',data=np.loadtxt(glob.glob(stims_folder+"*base*.1D")[0],dtype=int))
        stim_times.create_dataset(name='reg',data=np.loadtxt(glob.glob(stims_folder+"*reg*.1D")[0],dtype=int))
        stim_times.create_dataset(name='ireg',data=np.loadtxt(glob.glob(stims_folder+"*irr*.1D")[0],dtype=int))
        stim_times.create_dataset(name='pse',data=np.loadtxt(glob.glob(stims_folder+"*pse*.1D")[0],dtype=int))
        print(visit)
        for subj_id in subject_ids:
            subject = subj_id
            subject_file = list(filter(re.compile(".*"+subject).match,nifti_files))
            if subject_file:
                print("Preparing subject {}".format(subject))
                print("Loading time series...")
                time_series = masker.fit_transform(subject_file[0])
                try:
                    subject_group = output_file.require_group(name=subject)
                    visit_group = subject_group.require_group(name=visit)
                    visit_group.create_dataset(name='psc',data=time_series)
                except ValueError:
                    print("Duplicated subject!")
output_file.close()