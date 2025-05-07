
# get all the recordings on that day
# allocate destination folder and move the ephys folder on the server to Beast lab user
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
import docker
from datetime import datetime
import itertools

import scipy.io as sio
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
use_ks4 = sys.argv[5].lower() in ['true', '1', 't', 'y', 'yes']
use_ks3 = sys.argv[6].lower() in ['true', '1', 't', 'y', 'yes']
#base_folder = '/saleem/ibn-vision/DATA/SUBJECTS/'
#base_folder = '/home/masa/rds01/ibn-vision/DATA/SUBJECTS/'
base_folder = sys.argv[7]
save_folder = local_folder +save_date+'/'
# get all the recordings on that day
probe0_start_sample_fames = []
probe0_end_sample_frames = []
import os
import subprocess
subprocess.run('ulimit -n 10000',shell=True)
def sorting_key(s):
    return int(s.split('_g')[-1])

import pandas as pd
#grab recordings from the server to local machine (Beast)

job_kwargs = dict(n_jobs=32, chunk_duration='1s', progress_bar=True)

probes=[0]
for probe in probes:
    '''minor corrections to the folder path of files before moving the files to server'''
    #process to change all the folder paths in text and .json files on Beast to the server before uploading it to the server
    import os
    import glob

    # Define the folder list
    folder_list = [save_folder + 'probe'+str(probe)+'_preprocessed', 
                save_folder + 'probe'+str(probe)+'/waveform/',
                save_folder + 'probe'+str(probe)+'/sorters/',
                save_folder + 'probe'+str(probe)+'_motion/']

    # Initialize an empty list to store the paths of JSON files
    json_file_list = []
    temp_wh_files = []
    # Go through each folder in the folder list
    for folder in folder_list:
        # Recursively find all JSON files in the folder and its subfolders
        for json_file in glob.glob(os.path.join(folder, '**', '*.json'), recursive=True):
            # Append the found JSON file path to the list
            json_file_list.append(json_file)
    for folder in folder_list:
        # Recursively find all JSON files in the folder and its subfolders
        for temp_wh_file in glob.glob(os.path.join(folder, '**', 'temp_wh.dat'), recursive=True):
            # Append the found JSON file path to the list
            temp_wh_files.append(temp_wh_file)
    def replace_text(obj, to_replace, replace_with):
        if isinstance(obj, dict):
            return {k: replace_text(v, to_replace, replace_with) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [replace_text(elem, to_replace, replace_with) for elem in obj]
        elif isinstance(obj, str):
            return obj.replace(to_replace, replace_with)
        else:
            return obj
    import json

    for files in json_file_list:
        
        # open the JSON file and load the data
        with open(files, 'r') as f:
            data = json.load(f)
        
        # replace the text
        data = replace_text(data, save_folder[:-1], base_folder + mouse + '/ephys/' +save_date)
        
        # write the updated data back to the JSON file
        with open(files, 'w') as f:
            json.dump(data, f, indent=4)
            

    for files in temp_wh_files:
        os.remove(files)
    #move spikeinterface folder on Beast to the server

    import shutil
    import os

    ## NEW FUNCTION TO REPLACE shutil.copytree() now that we are using GVFS-mounted SMB share
    #import errno

    def copy_folder(src, dst):
        os.makedirs(dst, exist_ok=True)
        for root, dirs, files in os.walk(src):
            rel_path = os.path.relpath(root, src)
            dst_dir = os.path.join(dst, rel_path)
            os.makedirs(dst_dir, exist_ok=True)
            for file in files:
                src_file = os.path.join(root, file)
                dst_file = os.path.join(dst_dir, file)
                #try:
                shutil.copy2(src_file, dst_file)
                #except OSError as e:
                 #   if e.errno == errno.EOPNOTSUPP:
                  #      print(f"Skipping unsupported file: {src_file}")
                   # else:
                    #    raise


    ##
    #
    folders_to_move = ['probe'+str(probe),
                'probe'+str(probe)+'_motion/']

    for folder in folders_to_move:
        # construct the destination path
        destination = os.path.join(base_folder + mouse + '/ephys/' +save_date, folder)
        # copy the folder to the destination
        #shutil.copytree(save_folder + folder, destination)
        copy_folder(save_folder + folder, destination)
#
#remove all temmp files
#shutil.rmtree(save_folder)

print('All Done! Overall it took:')

print(datetime.now() - startTime)
print('to finish! Please move the files to the server as soon as you have time!')

sys.exit(0)
