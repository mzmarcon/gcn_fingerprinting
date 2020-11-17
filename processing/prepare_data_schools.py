import argparse
import numpy as np
import pandas as pd
import glob
import re
from utils import hdf5_handler
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure


parser = argparse.ArgumentParser()
parser.add_argument('--visits', default='visit1', #action='append',
                    help='visits for matrix calculation')
parser.add_argument('--in_rois', type=str, default='../../atlas/shen_1mm_268_parcellation_LPI.nii.gz',
                    help='path and filename for the parcellation to be used')
parser.add_argument('--subjects_ids', type=str, default='../subjects_ids.csv',
                    help='path and filename for the subjects ids file to be used')
parser.add_argument('--output_file', type=str, default='../../hdf5_files/shen_psc_task_schools.hdf5',
                    help='file to store the connectivity matrices output')
parser.add_argument('--nifti_type', type=str, default='psc',
                    help='data on the nifti file', choices=['errts','psc'])
parser.add_argument('--stims_folder', type=str, default='../stimuli/',
                    help='folder containing the stim times')

#-----------------------------------------------------------------------------------------------
args = parser.parse_args()
visits = ['visit1','visit2']#args.nifti_folder
in_rois = args.in_rois
subjects_ids_csv = args.subjects_ids
output_file = args.output_file
nifti_type = args.nifti_type
stims_folder = args.stims_folder

if not isinstance(visits,list):
    visits = [visits]

# Instance output file
output_file = hdf5_handler(output_file,'a')

# Load clinical data
subjects_ids = pd.read_csv(subjects_ids_csv,header=None)

# Create roi and correlation object
masker = NiftiLabelsMasker(labels_img=in_rois, standardize=True)
correlation_measure = ConnectivityMeasure(kind='correlation')

ids_nifts = subjects_ids.values.astype(str)
# ids_nifts = ids_nifts[ids_nifts != 'nan']


for visit in visits:
    if nifti_type == 'errts':
        nifti_files = glob.glob('../'+visit+'/*+tlrc.BRIK')
        print(visit)
        for subj_id in ids_nifts:
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
        nifti_files = glob.glob('../'+visit+'/PSC.*')
        stim_times = output_file.require_group(name='stim_times')
        stim_times.create_dataset(name='base',data=np.loadtxt(glob.glob(stims_folder+"*_base*.1D")[0],dtype=int))
        stim_times.create_dataset(name='reg',data=np.loadtxt(glob.glob(stims_folder+"*_reg*.1D")[0],dtype=int))
        stim_times.create_dataset(name='ireg',data=np.loadtxt(glob.glob(stims_folder+"*_ireg*.1D")[0],dtype=int))
        stim_times.create_dataset(name='pse',data=np.loadtxt(glob.glob(stims_folder+"*_pse*.1D")[0],dtype=int))
        print(visit)
        for subj_id in ids_nifts:
            subject = subj_id[0]
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
