
# get all the recordings on that day
# allocate destination folder and move the ephys folder on the server to Beast lab user
from pathlib import Path

import os
import shutil

import numpy as np
import glob
import spikeinterface.sorters
import spikeinterface.full as si
import  scipy.signal
import spikeinterface.extractors as se
import spikeinterface.comparison
import spikeinterface.exporters
import spikeinterface.curation
import spikeinterface.widgets 
import docker
from datetime import datetime
import itertools
import ast
import scipy.io as sio
import pandas as pd


startTime = datetime.now()
print('Start Time:' + startTime.strftime("%m/%d/%Y, %H:%M:%S"))
''' this section defines the animal and dates and fetch the recordings from the server to Beast'''
import sys
# The first command-line argument after the script name is the mouse identifier.
mouse = sys.argv[1]
# All command-line arguments after `mouse` and before `save_date` are considered dates.

# The last command-line argument is `save_date`.
save_date = sys.argv[2]
local_folder = sys.argv[3]
no_probe = sys.argv[4]
print(mouse)
print(save_date)
use_ks4 = sys.argv[5].lower() in ['true', '1', 't', 'y', 'yes']
use_ks3 = sys.argv[6].lower() in ['true', '1', 't', 'y', 'yes']
base_folder = '/saleem/ibn-vision/DATA/SUBJECTS/'
save_folder = local_folder + save_date +'/'

# Check g files to ignore are correct (tcat should always be ignored)
# Check if sys.argv[8] is empty
if len(sys.argv) > 7 and sys.argv[7]:
    g_files_to_ignore = ast.literal_eval(sys.argv[7])
else:
    g_files_to_ignore = []

# Print the result to verify
print(f"g_files_to_ignore: {g_files_to_ignore}")
# get all the recordings on that day

import os
import subprocess
subprocess.run('ulimit -n 4096',shell=True)
def sorting_key(s):
    return int(s.split('_g')[-1])

#grab recordings from the server to local machine (Beast)


job_kwargs = dict(n_jobs=32, chunk_duration='1s', progress_bar=True)

g_files_all = []
# iterate over all directories in source folder
acquisition_base_path = base_folder + mouse + '/ephys/' + save_date + '/*' + save_date
acquisition_folders = glob.glob(acquisition_base_path + '_*')
acquisition_list = sorted([int(folder.split('_')[-1]) for folder in acquisition_folders])
date_count = 0
for acquisition in acquisition_list:
    print('acquisition folder:',str(acquisition))
    date_count = date_count + 1
    for probe in range(int(no_probe)): 
        ephys_folder = base_folder + mouse + '/ephys/' + save_date +'/probe' + str(probe) + '_compressed_' + str(acquisition) + '.zarr'
        dst_folder = local_folder + save_date + '/'
        
        print('copying compressed data from:' + ephys_folder)

    
        destination = os.path.join(dst_folder, 'probe' + str(probe) + '_compressed_' + str(acquisition) + '.zarr')
        shutil.copytree(ephys_folder, destination)
        print('Start to copying files to local folder: ')
        print(datetime.now() - startTime)
        ''' read spikeglx recordings and preprocess them'''
        # Define a custom sorting key that extracts the number after 'g'

sys.exit(0)
