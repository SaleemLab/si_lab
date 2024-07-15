
# Just rename lf.bin files with t0 to tcat
from pathlib import Path

import os
import shutil

import numpy as np

import spikeinterface.sorters
import spikeinterface.full as si
import  scipy.signal
import spikeinterface.extractors as se
import spikeinterface.comparison
import spikeinterface.exporters
import spikeinterface.curation
import spikeinterface.widgets 
from datetime import datetime
import itertools
import glob
import scipy.io as sio
startTime = datetime.now()
print('Start Time:' + startTime.strftime("%m/%d/%Y, %H:%M:%S"))
''' this section defines the animal and dates and fetch the recordings from the server to Beast'''
mouse = 'M24016'
session = '20240706'
dates = ['20240706/20240706_0']
g_files_to_ignore = ['tcat','0_g6','0_g7','0_g8','0_g9']

save_date = '20240706'
base_folder = 'Z:/ibn-vision/DATA/SUBJECTS/'
save_base_folder = base_folder
save_folder = save_base_folder + mouse + '/ephys/' + save_date + '/'
# get all the recordings on that day
probe0_start_sample_fames = []
probe1_start_sample_frames = []
probe0_end_sample_frames = []
probe1_end_sample_frames = []
import os
import subprocess
subprocess.run('ulimit -n 4096',shell=True)
def sorting_key(s):
    return int(s.split('_g')[-1])

#grab recordings from the server to local machine (Beast)


job_kwargs = dict(n_jobs=8, chunk_duration='1s', progress_bar=True)


g_files_all = []
# iterate over all directories in source folder
date_count = 0
for date in dates:
    date_count = date_count + 1
    ephys_folder = base_folder + mouse + '/ephys/' + date +'/'
    dst_folder =  save_folder 
    ephys_folder = base_folder + mouse + '/ephys/' + date +'/'
    g_files = []
    # Step 1: Iterate over the list of files with tcat in the name
    tcat_pattern = os.path.join(ephys_folder, '**', '*t0.imec*.lf*')
    files_to_rename = glob.glob(tcat_pattern, recursive=True)
    for old_name in files_to_rename:
        # Step 4: Construct the new filename
        new_name = old_name.replace('t0', 'tcat')    
        # Step 5: Rename the file
        os.rename(old_name, new_name)
        print(f'Renamed {old_name} to {new_name}')
        