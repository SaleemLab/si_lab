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
import pandas as pd
import numpy as np
def save_spikes_to_csv(spikes,save_folder):
    unit_index = spikes['unit_index']
    segment_index = spikes['segment_index']
    sample_index = spikes['sample_index']
    spikes_df = pd.DataFrame({'unit_index':unit_index,'segment_index':segment_index,'sample_index':sample_index})
    spikes_df.to_csv(save_folder + 'spikes.csv',index=False)
#load mat file with merge suggestions
def replace_text(obj, to_replace, replace_with):
    if isinstance(obj, dict):
        return {k: replace_text(v, to_replace, replace_with) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_text(elem, to_replace, replace_with) for elem in obj]
    elif isinstance(obj, str):
        return obj.replace(to_replace, replace_with)
    else:
        return obj
base_folder = '/mnt/rds01/ibn-vision/DATA/SUBJECTS/'
mice = ['M24028','M24029','M24030','M24031','M24032', 'M24033', 'M24034','M24035','M24036','M23024','M23020','M23022','M23019']
all_dates = [['20240426','20240427'],['20240424','20240426'],['20240508', '20240509'],['20240510','20240511'],
             ['20240514','20240515'],['20240521','20240522'],['20240516','20240517'],['20240523','20240524'], 
             ['20240529','20240530','20240531'], ['20230823','20230824'], ['20230829','20230830'],
             ['20230831','20230901'],['20230902','20230903']]
mouse_number = 0
for mouse in mice:    
    dates = all_dates[mouse_number]
    mouse_number += 1
    for date in dates:
        ephys_folder = base_folder + mouse + '/ephys/' + date +'/'
        analysis_folder = base_folder + mouse + '/ephys/' + date + '/'
        save_folder = base_folder + mouse + '/ephys/' + date +'/'

        pathforprobe = base_folder + mouse + '/' + 'ephys' + '/' + date + '/' + mouse + '_' + date+'_g0/' + mouse +'_'+ date +'_g0_imec1'
        # Ensure the kilosort4_folder exists
        kilosort4_folder = os.path.join(ephys_folder, 'kilosort4')
        if not os.path.exists(kilosort4_folder):
            print(f"The directory {kilosort4_folder} does not exist.")
        else:
            # List all files in the kilosort4_folder
            for filename in os.listdir(kilosort4_folder):
                source_file_path = os.path.join(kilosort4_folder, filename)
                destination_file_path = os.path.join(ephys_folder, filename)
                
                # Move the file to the ephys_folder
                shutil.move(source_file_path, destination_file_path)
                print(f"Moved: {source_file_path} -> {destination_file_path}")

            # Optionally, remove the now-empty kilosort4_folder
            os.rmdir(kilosort4_folder)
            print(f"Removed empty directory: {kilosort4_folder}")
            import os.path
        if os.path.isdir(pathforprobe):
            print('Running dual probe pipeline')
                #awarkwardly, the path in the JSON file is different from the path in the folder
                #we need to replace the path in the JSON file to match the path in the folder
            import json
            file_list = [ephys_folder + 'probe0/sorters/kilosort4/spikeinterface_recording.json',
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
                data = replace_text(data, matching_strings, base_folder + mouse + '/ephys/')
                
                # write the updated data back to the JSON file
                with open(files, 'w') as f:
                    json.dump(data, f, indent=4)                
                    
            merge_suggestions = sio.loadmat(analysis_folder + 'probe0um_merge_suggestion.mat')
            probe0_sorting_ks3 = si.read_sorter_folder(ephys_folder + 'probe0/sorters/kilosort4/')
            probe0_we_ks3 = si.load_waveforms(ephys_folder + 'probe0/waveform/kilosort4/') 
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
            probe0_sorting_ks3_merged.save(folder = ephys_folder + 'probe0/sorters/kilosort4_merged/')

            merge_suggestions = sio.loadmat(analysis_folder + 'probe1um_merge_suggestion.mat')
            probe1_sorting_ks3 = si.read_sorter_folder(ephys_folder + 'probe1/sorters/kilosort4/')
            probe1_we_ks3 = si.load_waveforms(ephys_folder + 'probe1/waveform/kilosort4/') 
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
            probe1_sorting_ks3_merged.save(folder = ephys_folder + 'probe1/sorters/kilosort4_merged/')
            ''' Compute quality metrics on the extracted waveforms'''
            probe0_we_ks3_merged = si.create_sorting_analyzer(probe0_sorting_ks3_merged, probe0_preprocessed_corrected, 
                                    format = 'binary_folder',folder=save_folder +'probe0/waveform/kilosort4_merged',
                                    sparse = True,overwrite = True,
                                    **job_kwargs)

            probe1_we_ks3_merged = si.create_sorting_analyzer(probe1_sorting_ks3_merged, probe1_preprocessed_corrected, 
                                    format = 'binary_folder',folder=save_folder +'probe1/waveform/kilosort4_merged',
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
            probe0_ks4_spikes = np.load(save_folder + 'probe0/waveform/kilosort4_merged/sorting/spikes.npy')
            save_spikes_to_csv(probe0_ks4_spikes,save_folder + 'probe0/waveform/kilosort4_merged/sorting/')
            probe1_ks4_spikes = np.load(save_folder + 'probe1/waveform/kilosort4_merged/sorting/spikes.npy')
            save_spikes_to_csv(probe1_ks4_spikes,save_folder + 'probe1/waveform/kilosort4_merged/sorting/')
        
        else:
            print('Running single probe pipeline')
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
            file_list = [ephys_folder + 'probe0/sorters/kilosort4/spikeinterface_recording.json']
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
            probe0_sorting_ks3 = si.read_sorter_folder(ephys_folder + 'probe0/sorters/kilosort4/')
            probe0_we_ks3 = si.load_waveforms(ephys_folder + 'probe0/waveform/kilosort4/') 
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
            probe0_sorting_ks3_merged.save(folder = ephys_folder + 'probe0/sorters/kilosort4_merged/')
            
            ''' Compute quality metrics on the extracted waveforms'''
            probe0_we_ks3_merged = si.create_sorting_analyzer(probe0_sorting_ks3_merged, probe0_preprocessed_corrected, 
                                    format = 'binary_folder',folder=save_folder +'probe0/waveform/kilosort4_merged',
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
            probe0_ks4_spikes = np.load(save_folder + 'probe0/waveform/kilosort4_merged/sorting/spikes.npy')
            save_spikes_to_csv(probe0_ks4_spikes,save_folder + 'probe0/waveform/kilosort4_merged/sorting/')
        



        

