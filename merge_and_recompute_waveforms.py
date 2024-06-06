#we use UnitMatch to merge units that are similar and recompute the waveforms
#This script loads um_merge_suggestions.mat and merge units suggeted by UnitMatch
# Variable match_id: first column original id, and second column the id to merge with

import os
import numpy as np
import scipy.io as sio
from pathlib import Path
import platform
import os
import shutil
import argparse
import numpy as np
import matplotlib.pyplot as plt
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
#load mat file with merge suggestions
base_folder = '/mnt/rds01/ibn-vision/DATA/SUBJECTS/'
mouse = 'M23034'
dates = ['20230804','20230805','20230806','20230807']
ephys_folder = base_folder + mouse + '/ephys/' + dates[3] +'/'
analysis_folder = base_folder + mouse + '/analysis/' + dates[3] +'/'

#awarkwardly, the path in the JSON file is different from the path in the folder
#we need to replace the path in the JSON file to match the path in the folder
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
file_list = [ephys_folder + 'probe0/sorters/kilosort3/spikeinterface_recording.json',
             ephys_folder + 'probe1/sorters/kilosort3/spikeinterface_recording.json',
             ephys_folder + 'probe0/sorters/kilosort4/spikeinterface_recording.json',
             ephys_folder + 'probe1/sorters/kilosort4/spikeinterface_recording.json']
for files in file_list:
    
    # open the JSON file and load the data
    with open(files, 'r') as f:
        data = json.load(f)
        # Initialize an empty list to store the matching strings
    matching_strings = []

    folder_path = data['kwargs']['folder_path']
    # Check if the string starts with '/home/' and ends with 'temp_data/'
    start = folder_path.find('/home/')
    end = folder_path.find('temp_data/')
    matching_strings = folder_path[start:end] + 'temp_data/'

    # replace the text
    data = replace_text(data, matching_strings, ephys_folder)
    
    # write the updated data back to the JSON file
    with open(files, 'w') as f:
        json.dump(data, f, indent=4)
        
        
merge_suggestions = sio.loadmat(analysis_folder + 'probe0um_merge_suggestion.mat')
match_ids = merge_suggestions['match_ids']
merge_ids = match_ids[:,1]
#load recordings
probe0_sorting = si.read_sorter_folder(ephys_folder + 'probe0/sorters/kilosort4/')
probe0_we = si.load_waveforms(ephys_folder + 'probe0/waveform/kilosort4/') 
#merge units


cs_probe0 = si.CurationSorting(parent_sorting=probe0_sorting)
unique_ids = np.unique(match_ids)
for id in unique_ids:
    id_count = np.count_nonzero(match_id == id)
    if id_count > 1:
        units = np.where(match_id == id)
        cs_probe0.MergeUnitsSorting(match_id(units))