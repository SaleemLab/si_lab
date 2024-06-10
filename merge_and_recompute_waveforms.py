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
save_folder = base_folder + mouse + '/ephys/' + dates[3] +'/'
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
original_ids = match_ids[:,0]
#load recordings
job_kwargs = dict(n_jobs=32, chunk_duration='1s', progress_bar=True)
probe0_sorting_ks4 = si.read_sorter_folder(ephys_folder + 'probe0/sorters/kilosort4/')
probe0_we_ks4 = si.load_waveforms(ephys_folder + 'probe0/waveform/kilosort4/') 
probe0_preprocessed_corrected = si.load_extractor(ephys_folder + 'probe0_preprocessed/')



cs_probe0 = si.CurationSorting(parent_sorting=probe0_sorting_ks4)
unique_ids = np.unique(merge_ids)
for id in unique_ids:
    id_count = np.count_nonzero(merge_ids == id)
    if id_count > 1:
        units = np.where(merge_ids == id)
        units_index = (units[0]-1,)
        cs_probe0.merge(original_ids[units_index])
        
probe0_sorting_ks4_merged = cs_probe0.sorting
probe0_sorting_ks4_merged.save(folder = ephys_folder + 'probe0/sorters/kilosort4_merged/')

job_kwargs = dict(n_jobs=32, chunk_duration='1s', progress_bar=True)
probe1_sorting_ks4 = si.read_sorter_folder(ephys_folder + 'probe1/sorters/kilosort4/')
probe1_we_ks4 = si.load_waveforms(ephys_folder + 'probe1/waveform/kilosort4/') 
probe1_preprocessed_corrected = si.load_extractor(ephys_folder + 'probe1_preprocessed/')
#merge units


cs_probe1 = si.CurationSorting(parent_sorting=probe1_sorting_ks4)
unique_ids = np.unique(merge_ids)
for id in unique_ids:
    id_count = np.count_nonzero(merge_ids == id)
    if id_count > 1:
        units = np.where(merge_ids == id)
        units_index = (units[0]-1,)
        cs_probe1.merge(original_ids[units_index])
        
probe1_sorting_ks4_merged = cs_probe1.sorting
probe1_sorting_ks4_merged.save(folder = ephys_folder + 'probe1/sorters/kilosort4_merged/')
''' Compute quality metrics on the extracted waveforms'''
probe0_we_ks4_merged = si.create_sorting_analyzer(probe0_preprocessed_corrected, probe0_sorting_ks4_merged, folder=save_folder +'probe0/waveform/kilosort4_merged',
                        sparse=True, max_spikes_per_unit=500, ms_before=1.5,ms_after=2.,
                        **job_kwargs)

probe1_we_ks4_merged = si.create_sorting_analyzer(probe1_preprocessed_corrected, probe1_sorting_ks4_merged, folder=save_folder +'probe1/waveform/kilosort4_merged',
                        sparse=True, max_spikes_per_unit=500, ms_before=1.5,ms_after=2.,
                        **job_kwargs)

probe0_we_ks4_merged.compute('waveforms')
probe1_we_ks4_merged.compute('waveforms')

probe0_we_ks4_merged.compute('template_metrics')
probe1_we_ks4_merged.compute('template_metrics')

probe0_we_ks4_merged.compute('noise_levels')
probe1_we_ks4_merged.compute('noise_levels')

probe0_we_ks4_merged.compute('principal_components',**job_kwargs)
probe1_we_ks4_merged.compute('principal_components',**job_kwargs)

probe0_we_ks4_merged.compute('template_similarity')
probe1_we_ks4_merged.compute('template_similarity')

probe0_we_ks4_merged.compute('correlograms')
probe1_we_ks4_merged.compute('correlograms')

probe0_we_ks4_merged.compute('spike_amplitudes',**job_kwargs)
probe1_we_ks4_merged.compute('spike_amplitudes',**job_kwargs)

probe0_we_ks4_merged.compute('isi_histograms')
probe1_we_ks4_merged.compute('isi_histograms')

qm_list = si.get_default_qm_params()
print('The following quality metrics are computed:')
print(qm_list)
probe0_we_ks4_merged.compute('quality_metrics', qm_params=qm_list,**job_kwargs)
probe1_we_ks4_merged.compute('quality_metrics', qm_params=qm_list,**job_kwargs)
