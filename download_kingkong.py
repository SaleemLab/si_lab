
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
dates = sys.argv[2].split(',')   # This captures all dates as a list.
# The last command-line argument is `save_date`.
save_date = sys.argv[3]
local_folder = sys.argv[4]
no_probe = sys.argv[5]
print(mouse)
print('acquisition folder: ',dates)
print(save_date)
use_ks4 = sys.argv[6].lower() in ['true', '1', 't', 'y', 'yes']
use_ks3 = sys.argv[7].lower() in ['true', '1', 't', 'y', 'yes']
base_folder = '/saleem/ibn-vision/DATA/SUBJECTS/'
save_folder = local_folder + save_date +'/'

# Check g files to ignore are correct (tcat should always be ignored)
g_files_to_ignore = ['tcat','0_g6','0_g7','0_g8','0_g9','lf.bin','if.meta']
#g_files_to_ignore = sys.argv[8]
print(g_files_to_ignore)
# get all the recordings on that day

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

print(dates)
g_files_all = []
# iterate over all directories in source folder
date_count = 0
for date in dates:
    print('acquisition folder:',date)
    date_count = date_count + 1
    ephys_folder = base_folder + mouse + '/ephys/' + date +'/'
    dst_folder = local_folder + date + '/'
    ephys_folder = base_folder + mouse + '/ephys/' + date +'/'
    g_files = []

    # Rename all tcat files to t0 if they exist
    tcat_pattern = os.path.join(ephys_folder,'**','*tcat.imec*.lf*')
    files_to_rename = glob.glob(tcat_pattern, recursive=True)
    # Step 1: Iterate over the list of files with tcat in the name
    for old_name in files_to_rename:
        # Step 2: Construct the new filename (REMEMBER to switch the name back to tcat)
        new_name = old_name.replace('tcat', 't0')
        
        # Step 3: Rename the file
        os.rename(old_name, new_name)
        print(f'Renamed {old_name} to {new_name}')

    print('copying ephys data from:' + ephys_folder)
    for dirname in os.listdir(ephys_folder):
        # ignore some folders or files includiing tcat and other G files (such as checkerboard recordings)
        if any(ignore_str in dirname for ignore_str in g_files_to_ignore):
            continue
    #     # check if '_g' is in the directory name
    #     #only grab recording folders - there might be some other existing folders for analysis or sorted data
        if '_g' in dirname:
    #         # construct full directory path
            g_files.append(dirname)
            source = os.path.join(ephys_folder, dirname)
            destination = os.path.join(dst_folder, dirname)
            # copy the directory to the destination folder
            shutil.copytree(source, destination)
    print('Start to copying files to local folder: ')
    print(datetime.now() - startTime)
    ''' read spikeglx recordings and preprocess them'''
    # Define a custom sorting key that extracts the number after 'g'

    # Sort the list using the custom sorting key
    g_files = sorted(g_files, key=sorting_key)
    g_files_all = g_files_all + g_files
    print(g_files)
    print('all g files:',g_files_all) 
    # stream_names, stream_ids = si.get_neo_streams('spikeglx',dst_folder)
    # print(stream_names)
    # print(stream_ids)


g_files_all = []
# iterate over all directories in source folder
date_count = 0
for date in dates:
    date_count = date_count + 1
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
        
sys.exit(0)
