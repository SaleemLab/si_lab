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
mouse = 'M23037'
dates = ['20230810','20230811','20230812','20230813']
for date in dates:
    ephys_folder = base_folder + mouse + '/ephys/' + date +'/'
    analysis_folder = base_folder + mouse + '/analysis/' + date +'/'
    save_folder = base_folder + mouse + '/ephys/' + date +'/'
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
                ephys_folder + 'probe1/sorters/kilosort3/spikeinterface_recording.json']
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
        data = replace_text(data, matching_strings, base_folder + mouse + '/ephys/')
        
        # write the updated data back to the JSON file
        with open(files, 'w') as f:
            json.dump(data, f, indent=4)
            
            
    merge_suggestions = sio.loadmat(analysis_folder + 'probe0um_merge_suggestion.mat')
    probe0_sorting_ks3 = si.read_sorter_folder(ephys_folder + 'probe0/sorters/kilosort3/')
    probe0_we_ks3 = si.load_waveforms(ephys_folder + 'probe0/waveform/kilosort3/') 
    probe0_preprocessed_corrected = si.load_extractor(ephys_folder + 'probe0_preprocessed/')
    match_ids = merge_suggestions['match_ids']
    merge_ids = match_ids[:,1] - 1
    cs_probe0 = si.CurationSorting(probe0_sorting_ks3)
    unique_ids = np.unique(merge_ids)
    original_ids = probe0_sorting_ks3.get_unit_ids()
    for id in unique_ids:
        id_count = np.count_nonzero(merge_ids == id)
        if id_count > 1:
            unit_index = merge_ids == id
            cs_probe0.merge(original_ids[unit_index])
    #load recordings
    job_kwargs = dict(n_jobs=32, chunk_duration='1s', progress_bar=True)

    probe0_sorting_ks3_merged = cs_probe0.sorting
    probe0_sorting_ks3_merged.save(folder = ephys_folder + 'probe0/sorters/kilosort3_merged/')

    merge_suggestions = sio.loadmat(analysis_folder + 'probe1um_merge_suggestion.mat')
    probe1_sorting_ks3 = si.read_sorter_folder(ephys_folder + 'probe1/sorters/kilosort3/')
    probe1_we_ks3 = si.load_waveforms(ephys_folder + 'probe1/waveform/kilosort3/') 
    probe1_preprocessed_corrected = si.load_extractor(ephys_folder + 'probe1_preprocessed/')
    match_ids = merge_suggestions['match_ids']
    merge_ids = match_ids[:,1] - 1
    cs_probe1 = si.CurationSorting(probe1_sorting_ks3)
    unique_ids = np.unique(merge_ids)
    original_ids = probe1_sorting_ks3.get_unit_ids()
    for id in unique_ids:
        id_count = np.count_nonzero(merge_ids == id)
        if id_count > 1:
            unit_index = merge_ids == id
            cs_probe1.merge(original_ids[unit_index])
    #load recordings
    job_kwargs = dict(n_jobs=32, chunk_duration='1s', progress_bar=True)

    probe1_sorting_ks3_merged = cs_probe1.sorting
    probe1_sorting_ks3_merged.save(folder = ephys_folder + 'probe1/sorters/kilosort3_merged/')
    ''' Compute quality metrics on the extracted waveforms'''
    probe0_we_ks3_merged = si.create_sorting_analyzer(probe0_sorting_ks3_merged, probe0_preprocessed_corrected, 
                            format = 'binary_folder',folder=save_folder +'probe0/waveform/kilosort3_merged',
                            sparse = True,overwrite = True,
                            **job_kwargs)

    probe1_we_ks3_merged = si.create_sorting_analyzer(probe1_sorting_ks3_merged, probe1_preprocessed_corrected, 
                            format = 'binary_folder',folder=save_folder +'probe1/waveform/kilosort3_merged',
                            sparse = True,overwrite = True,
                            **job_kwargs)
    extensions = ['templates', 'template_metrics', 'noise_levels', 'template_similarity', 'correlograms', 'isi_histograms']
    probe0_we_ks3_merged.compute('random_spikes')
    probe1_we_ks3_merged.compute('random_spikes')

    probe0_we_ks3_merged.compute('waveforms',ms_before=1.0, ms_after=2.0,**job_kwargs)
    probe1_we_ks3_merged.compute('waveforms',ms_before=1.0, ms_after=2.0,**job_kwargs)
    probe0_we_ks3_merged.compute(extensions,**job_kwargs)
    probe1_we_ks3_merged.compute(extensions,**job_kwargs)


    probe0_we_ks3_merged.compute('spike_amplitudes',**job_kwargs)
    probe1_we_ks3_merged.compute('spike_amplitudes',**job_kwargs)



    qm_list = si.get_default_qm_params()
    print('The following quality metrics are computed:')
    print(qm_list)
    probe0_we_ks3_merged.compute('quality_metrics', qm_params=qm_list,**job_kwargs)
    probe1_we_ks3_merged.compute('quality_metrics', qm_params=qm_list,**job_kwargs)

    import pandas as pd
    import numpy as np
    def save_spikes_to_csv(spikes,save_folder):
        unit_index = spikes['unit_index']
        segment_index = spikes['segment_index']
        sample_index = spikes['sample_index']
        spikes_df = pd.DataFrame({'unit_index':unit_index,'segment_index':segment_index,'sample_index':sample_index})
        spikes_df.to_csv(save_folder + 'spikes.csv',index=False)
        
    for probe in [0,1]:
        probe0_ks3_spikes = np.load(save_folder + 'probe'+str(probe)+'/waveform/kilosort3_merged/sorting/spikes.npy')
        save_spikes_to_csv(probe0_ks3_spikes,save_folder + 'probe'+str(probe)+'/waveform/kilosort3_merged/sorting/')
#we use UnitMatch to merge units that are similar and recompute the waveforms
#This script loads um_merge_suggestions.mat and merge units suggeted by UnitMatch
# Variable match_id: first column original id, and second column the id to merge with


#load mat file with merge suggestions
base_folder = '/mnt/rds01/ibn-vision/DATA/SUBJECTS/'
mouse = 'M23038'
dates = ['20230816','20230817']
for date in dates:
    ephys_folder = base_folder + mouse + '/ephys/' + date +'/'
    analysis_folder = base_folder + mouse + '/analysis/' + date +'/'
    save_folder = base_folder + mouse + '/ephys/' + date +'/'
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
                ephys_folder + 'probe1/sorters/kilosort3/spikeinterface_recording.json']
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
        data = replace_text(data, matching_strings, base_folder + mouse + '/ephys/')
        
        # write the updated data back to the JSON file
        with open(files, 'w') as f:
            json.dump(data, f, indent=4)
            
            
    merge_suggestions = sio.loadmat(analysis_folder + 'probe0um_merge_suggestion.mat')
    probe0_sorting_ks3 = si.read_sorter_folder(ephys_folder + 'probe0/sorters/kilosort3/')
    probe0_we_ks3 = si.load_waveforms(ephys_folder + 'probe0/waveform/kilosort3/') 
    probe0_preprocessed_corrected = si.load_extractor(ephys_folder + 'probe0_preprocessed/')
    match_ids = merge_suggestions['match_ids']
    merge_ids = match_ids[:,1] - 1
    cs_probe0 = si.CurationSorting(probe0_sorting_ks3)
    unique_ids = np.unique(merge_ids)
    original_ids = probe0_sorting_ks3.get_unit_ids()
    for id in unique_ids:
        id_count = np.count_nonzero(merge_ids == id)
        if id_count > 1:
            unit_index = merge_ids == id
            cs_probe0.merge(original_ids[unit_index])
    #load recordings
    job_kwargs = dict(n_jobs=32, chunk_duration='1s', progress_bar=True)

    probe0_sorting_ks3_merged = cs_probe0.sorting
    probe0_sorting_ks3_merged.save(folder = ephys_folder + 'probe0/sorters/kilosort3_merged/')

    merge_suggestions = sio.loadmat(analysis_folder + 'probe1um_merge_suggestion.mat')
    probe1_sorting_ks3 = si.read_sorter_folder(ephys_folder + 'probe1/sorters/kilosort3/')
    probe1_we_ks3 = si.load_waveforms(ephys_folder + 'probe1/waveform/kilosort3/') 
    probe1_preprocessed_corrected = si.load_extractor(ephys_folder + 'probe1_preprocessed/')
    match_ids = merge_suggestions['match_ids']
    merge_ids = match_ids[:,1] - 1
    cs_probe1 = si.CurationSorting(probe1_sorting_ks3)
    unique_ids = np.unique(merge_ids)
    original_ids = probe1_sorting_ks3.get_unit_ids()
    for id in unique_ids:
        id_count = np.count_nonzero(merge_ids == id)
        if id_count > 1:
            unit_index = merge_ids == id
            cs_probe1.merge(original_ids[unit_index])
    #load recordings
    job_kwargs = dict(n_jobs=32, chunk_duration='1s', progress_bar=True)

    probe1_sorting_ks3_merged = cs_probe1.sorting
    probe1_sorting_ks3_merged.save(folder = ephys_folder + 'probe1/sorters/kilosort3_merged/')
    ''' Compute quality metrics on the extracted waveforms'''
    probe0_we_ks3_merged = si.create_sorting_analyzer(probe0_sorting_ks3_merged, probe0_preprocessed_corrected, 
                            format = 'binary_folder',folder=save_folder +'probe0/waveform/kilosort3_merged',
                            sparse = True,overwrite = True,
                            **job_kwargs)

    probe1_we_ks3_merged = si.create_sorting_analyzer(probe1_sorting_ks3_merged, probe1_preprocessed_corrected, 
                            format = 'binary_folder',folder=save_folder +'probe1/waveform/kilosort3_merged',
                            sparse = True,overwrite = True,
                            **job_kwargs)
    extensions = ['templates', 'template_metrics', 'noise_levels', 'template_similarity', 'correlograms', 'isi_histograms']
    probe0_we_ks3_merged.compute('random_spikes')
    probe1_we_ks3_merged.compute('random_spikes')

    probe0_we_ks3_merged.compute('waveforms',ms_before=1.0, ms_after=2.0,**job_kwargs)
    probe1_we_ks3_merged.compute('waveforms',ms_before=1.0, ms_after=2.0,**job_kwargs)
    probe0_we_ks3_merged.compute(extensions,**job_kwargs)
    probe1_we_ks3_merged.compute(extensions,**job_kwargs)


    probe0_we_ks3_merged.compute('spike_amplitudes',**job_kwargs)
    probe1_we_ks3_merged.compute('spike_amplitudes',**job_kwargs)



    qm_list = si.get_default_qm_params()
    print('The following quality metrics are computed:')
    print(qm_list)
    probe0_we_ks3_merged.compute('quality_metrics', qm_params=qm_list,**job_kwargs)
    probe1_we_ks3_merged.compute('quality_metrics', qm_params=qm_list,**job_kwargs)

#load mat file with merge suggestions
base_folder = '/mnt/rds01/ibn-vision/DATA/SUBJECTS/'
mouse = 'M23031'
dates = ['20230711','20230712','20230713','20230714']
for date in dates:
    ephys_folder = base_folder + mouse + '/ephys/' + date +'/'
    analysis_folder = base_folder + mouse + '/analysis/' + date +'/'
    save_folder = base_folder + mouse + '/ephys/' + date +'/'
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
    file_list = [ephys_folder + 'probe0/sorters/kilosort3/spikeinterface_recording.json']
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
        data = replace_text(data, matching_strings, base_folder + mouse + '/ephys/')
        
        # write the updated data back to the JSON file
        with open(files, 'w') as f:
            json.dump(data, f, indent=4)
            
            
    merge_suggestions = sio.loadmat(analysis_folder + 'probe0um_merge_suggestion.mat')
    probe0_sorting_ks3 = si.read_sorter_folder(ephys_folder + 'probe0/sorters/kilosort3/')
    probe0_we_ks3 = si.load_waveforms(ephys_folder + 'probe0/waveform/kilosort3/') 
    probe0_preprocessed_corrected = si.load_extractor(ephys_folder + 'probe0_preprocessed/')
    match_ids = merge_suggestions['match_ids']
    merge_ids = match_ids[:,1] - 1
    cs_probe0 = si.CurationSorting(probe0_sorting_ks3)
    unique_ids = np.unique(merge_ids)
    original_ids = probe0_sorting_ks3.get_unit_ids()
    for id in unique_ids:
        id_count = np.count_nonzero(merge_ids == id)
        if id_count > 1:
            unit_index = merge_ids == id
            cs_probe0.merge(original_ids[unit_index])
    #load recordings
    job_kwargs = dict(n_jobs=32, chunk_duration='1s', progress_bar=True)

    probe0_sorting_ks3_merged = cs_probe0.sorting
    probe0_sorting_ks3_merged.save(folder = ephys_folder + 'probe0/sorters/kilosort3_merged/')

    
    ''' Compute quality metrics on the extracted waveforms'''
    probe0_we_ks3_merged = si.create_sorting_analyzer(probe0_sorting_ks3_merged, probe0_preprocessed_corrected, 
                            format = 'binary_folder',folder=save_folder +'probe0/waveform/kilosort3_merged',
                            sparse = True,overwrite = True,
                            **job_kwargs)


    extensions = ['templates', 'template_metrics', 'noise_levels', 'template_similarity', 'correlograms', 'isi_histograms']
    probe0_we_ks3_merged.compute('random_spikes')


    probe0_we_ks3_merged.compute('waveforms',ms_before=1.0, ms_after=2.0,**job_kwargs)

    probe0_we_ks3_merged.compute(extensions,**job_kwargs)



    probe0_we_ks3_merged.compute('spike_amplitudes',**job_kwargs)




    qm_list = si.get_default_qm_params()
    print('The following quality metrics are computed:')
    print(qm_list)
    probe0_we_ks3_merged.compute('quality_metrics', qm_params=qm_list,**job_kwargs)

