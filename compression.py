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
import sys
import glob
import ast

from wavpack_numcodecs import WavPack
compressor_wv = WavPack(level=3, bps=None)
# The first command-line argument after the script name is the mouse identifier.
mouse = sys.argv[1]
# All command-line arguments after `mouse` and before `save_date` are considered dates.
acquisition_list = ast.literal_eval(sys.argv[2])   # This captures all acquisitions as a list.
# The last command-line argument is `save_date`.
print(sys.argv[3])
save_date_list = ast.literal_eval(sys.argv[3])
local_folder = sys.argv[4]
no_probe = sys.argv[5]

no_dates = len(save_date_list)

acquisition_folders = []
date_count = 0
for save_date_tmp in save_date_list:
    save_date = str(save_date_tmp[0])
    acquisitions = acquisition_list[date_count]
    for acquisition in acquisitions:
        acquisition_folders.append(local_folder + mouse + '/ephys/' + save_date + '/' + save_date + '_' + str(acquisition)) 

    for probe in range(int(no_probe)):
        for acquisition_folder in acquisition_folders:
                # Rename all tcat files to t0 if they exist
            tcat_pattern = os.path.join(acquisition_folder,'**','*tcat.imec*.lf*')
            files_to_rename = glob.glob(tcat_pattern, recursive=True)
            # Step 1: Iterate over the list of files with tcat in the name
            for old_name in files_to_rename:
                # Step 2: Construct the new filename (REMEMBER to switch the name back to tcat)
                new_name = old_name.replace('tcat', 't0')
                
                # Step 3: Rename the file
                os.rename(old_name, new_name)
                print(f'Renamed {old_name} to {new_name}')
            
            recording_raw = si.read_spikeglx(acquisition_folder,stream_name='imec' + str(probe) + '.ap')
            print(recording_raw)
            compression_folder = local_folder + save_date + '/probe' + str(probe) + '_compressed' + acquisition_folder[-2:]
            print('compressing to folder: '+compression_folder)
            raw_compressed = recording_raw.save(format="zarr", folder=compression_folder, compressor=compressor_wv, n_jobs=16, chunk_duration="1s")
            tcat_pattern = os.path.join(acquisition_folder, '**', '*t0.imec*.lf*')
            files_to_rename = glob.glob(tcat_pattern, recursive=True)
            for old_name in files_to_rename:
                new_name = old_name.replace('t0', 'tcat')    
                os.rename(old_name, new_name)
                print(f'Renamed {old_name} to {new_name}')
    date_count = date_count + 1
