#import the necessary packages
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
mouse = 'M23022'
date = '20230831'


job_kwargs = dict(n_jobs=32, chunk_duration='1s', progress_bar=True)
base_folder = '/mnt/rds01/ibn-vision/DATA/SUBJECTS/'
dst_folder = "/home/saleem_lab/spikeinterface_sorting/temp_data/" + date + '/'
ephys_folder = base_folder + mouse + '/ephys/' + date +'/'

probe0_sorting_ks4 = si.run_sorter(sorter_name= 'kilosort4',recording=probe0_preprocessed_corrected,output_folder=dst_folder+'probe0/sorters/kilosort4/',docker_image=True,do_correction=False)
    #job_list = [
    # {'sorter_name':'kilosort2_5','recording':probe0_preprocessed_corrected,'output_folder':dst_folder+'/probe0/sorters/kilosort2_5/','docker_image':'spikeinterface/kilosort2_5-compiled-base','do_correction':False},
    #  {'sorter_name':'kilosort3','recording':probe0_preprocessed_corrected,'output_folder':dst_folder+'/probe0/sorters/kilosort3/','docker_image':True,'do_correction':False},
    # {'sorter_name':'mountainsort5','recording':probe0_preprocessed_corrected,'output_folder':dst_folder+'/probe0/sorters/mountainsort5/','docker_image':True},
    #    {'sorter_name':'kilosort2_5','recording':probe1_preprocessed_corrected,'output_folder':dst_folder+'/probe1/sorters/kilosort2_5/','docker_image':'spikeinterface/kilosort2_5-compiled-base','do_correction':False},
    #  {'sorter_name':'kilosort3','recording':probe1_preprocessed_corrected,'output_folder':dst_folder+'/probe1/sorters/kilosort3/','docker_image':True,'do_correction':False},
    # {'sorter_name':'mountainsort5','recording':probe1_preprocessed_corrected,'output_folder':dst_folder+'/probe1/sorters/mountainsort5/','docker_image':True},
    #]
    #run sorters in parallel
    #sortings = si.run_sorter_jobs(job_list = job_list,engine = 'joblib',engine_kwargs = {'n_jobs': 2})
    #remove duplicates
probe0_sorting_ks4 = si.remove_duplicated_spikes(sorting = probe0_sorting_ks4, censored_period_ms=0.3,method='keep_first')
print(probe0_sorting_ks4)
print('Start to all sorting done:')
print(datetime.now() - startTime)

import pandas as pd
probe0_segment_frames = pd.DataFrame({'segment_info':g_files,'segment start frame': probe0_start_sample_frames, 'segment end frame': probe0_end_sample_frames})
probe0_segment_frames.to_csv(dst_folder+'probe0/sorters/segment_frames.csv', index=False)
probe0_preprocessed_corrected = si.load_extractor(dst_folder + 'probe0_preprocessed')
#probe1_preprocessed_corrected = si.load_extractor(dst_folder + 'probe1_preprocessed')
probe0_sorting_ks4 = spikeinterface.sorters.read_sorter_foldexiter(dst_folder+'/probe0/sorters/kilosort4/', register_recording=True, sorting_info=True, raise_error=True)
#probe1_sorting_ks4 = spikeinterface.sorters.read_sorter_folder(dst_folder+'/probe1/sorters/kilosort4/', register_recording=True, sorting_info=True, raise_error=True)
#probe1_sorting_ks2_5 = spikeinterface.sorters.read_sorter_folder(dst_folder+'/probe1/sorters/kilosort2_5/', register_recording=True, sorting_info=True, raise_error=True)
probe0_we_ks4 = si.extract_waveforms(probe0_preprocessed_corrected, probe0_sorting_ks4, folder=dst_folder +'probe0/waveform/kilosort4',
                        sparse=True, max_spikes_per_unit=500, ms_before=1.5,ms_after=2.,
                        **job_kwargs)
#probe1_we_ks4 = si.extract_waveforms(probe1_preprocessed_corrected, probe1_sorting_ks4, folder=dst_folder +'probe1/waveform/kilosort4',
                        #sparse=True, max_spikes_per_unit=500, ms_before=1.5,ms_after=2.,
                        #**job_kwargs)

''' Compute quality metrics on the extracted waveforms'''
template_metric_probe0_ks4 = si.compute_template_metrics(probe0_we_ks4)
#template_metric_probe1_ks4 = si.compute_template_metrics(probe1_we_ks4)

noise_levels_probe0_ks4 = si.compute_noise_levels(probe0_we_ks4)
#noise_levels_probe1_ks4 = si.compute_noise_levels(probe1_we_ks4)

PCA_probe0_ks4 = si.compute_principal_components(probe0_we_ks4,**job_kwargs)
#PCA_probe1_ks4 = si.compute_principal_components(probe1_we_ks4,**job_kwargs)


template_similarity_probe0_ks4 = si.compute_template_similarity(probe0_we_ks4)
#template_similarity_probe1_ks4 = si.compute_template_similarity(probe1_we_ks4)

correlograms_probe0_ks4 = si.compute_correlograms(probe0_we_ks4)
#correlograms_probe1_ks4 = si.compute_correlograms(probe1_we_ks4)

amplitudes_probe0_ks4 = si.compute_spike_amplitudes(probe0_we_ks4,**job_kwargs)
#amplitudes_probe1_ks4 = si.compute_spike_amplitudes(probe1_we_ks4,**job_kwargs)


isi_histograms_probe0_ks4 = si.compute_isi_histograms(probe0_we_ks4)
#isi_histograms_probe1_ks4 = si.compute_isi_histograms(probe1_we_ks4)

qm_list = si.get_quality_metric_list()
print('The following quality metrics are computed:')
print(qm_list)
probe0_ks4_metrics = si.compute_quality_metrics(probe0_we_ks4, metric_names=qm_list,**job_kwargs)
#probe1_ks4_metrics = si.compute_quality_metrics(probe1_we_ks4, metric_names=qm_list,**job_kwargs)

'''minor corrections to the folder path of files before moving the files to server'''
#process to change all the folder paths in text and .json files on Beast to the server before uploading it to the server
file_list = [dst_folder + "probe0_preprocessed/provenance.json",
            #dst_folder + "probe1_preprocessed/provenance.json",
            dst_folder + "probe0/waveform/kilosort4/recording.json",
            dst_folder + "probe0/waveform/kilosort4/sorting.json",
            dst_folder + "probe0/sorters/kilosort4/in_container_sorting/provenance.json",
            dst_folder + "probe0/sorters/kilosort4/in_container_sorting/si_folder.json"]
            #dst_folder + "probe1/waveform/kilosort4/recording.json",
            #dst_folder + "probe1/waveform/kilosort4/sorting.json",
            #dst_folder + "probe1/sorters/kilosort4/in_container_sorting/provenance.json",
            #dst_folder + "probe1/sorters/kilosort4/in_container_sorting/si_folder.json"]

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

for files in file_list:
    
    # open the JSON file and load the data
    with open(files, 'r') as f:
        data = json.load(f)
    
    # replace the text
    data = replace_text(data, dst_folder[:-1], ephys_folder[:-1])
    
    # write the updated data back to the JSON file
    with open(files, 'w') as f:
        json.dump(data, f, indent=4)

#move spikeinterface folder on Beast to the server

import shutil
import os

folders_to_move = ['probe0',
                #'probe1',
                'probe0_preprocessed']
                #'probe1_preprocessed']
##
#
for folder in folders_to_move:
    # construct the destination path
    destination = os.path.join(ephys_folder, folder)
    # copy the folder to the destination
    shutil.copytree(dst_folder+folder, destination)
#
#remove all temmp files
#    shutil.rmtree(dst_folder)

print('All Done! Overall it took:')

