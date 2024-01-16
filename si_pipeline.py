# a script to run the spike interface on Diao's data with visualisation


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
startTime = datetime.now()
print('Start Time:' + startTime.strftime("%m/%d/%Y, %H:%M:%S"))
''' this section defines the animal and dates and fetch the recordings from the server to Beast'''

#grab recordings from the server to local machine (Beast)
base_folder = '/mnt/rds01/ibn-vision/DATA/SUBJECTS/'
mouse = 'M23028'
date = '20230706'
# get all the recordings on that day
ephys_folder = base_folder + mouse + '/ephys/' + date +'/'
# allocate destination folder and move the ephys folder on the server to Beast lab user
dst_folder = "/home/lab/spikeinterface_sorting/temp_data"

print('copying ephys data from:' + ephys_folder)
# iterate over all directories in source folder
for dirname in os.listdir(ephys_folder):
    # check if '_g' is in the directory name
    #only grab recording folders - there might be some other existing folders for analysis or sorted data
    if '_g' in dirname:
        # construct full directory path
        source = os.path.join(ephys_folder, dirname)
        destination = os.path.join(dst_folder, dirname)
        # copy the directory to the destination folder
        shutil.copytree(source, destination)
print('Start to copying files to Beast:')
print(datetime.now() - startTime)
''' read spikeglx recordings and preprocess them'''

stream_names, stream_ids = si.get_neo_streams('spikeglx',dst_folder)
print(stream_names)
print(stream_ids)
#load first probe from beast folder - MEC probe for Diao
probe0_raw = si.read_spikeglx(dst_folder,stream_name='imec0.ap')
print(probe0_raw)
#Load second probe - V1 probe
probe1_raw = si.read_spikeglx(dst_folder,stream_name = 'imec1.ap')
print(probe1_raw)

#several preprocessing steps and concatenation of the recordings
#highpass filter - threhsold at 300Hz
probe0_highpass = si.highpass_filter(probe0_raw,freq_min=300.)
#detect bad channels
bad_channel_ids, channel_labels = si.detect_bad_channels(probe0_highpass)
#remove bad channels if wanted
probe0_remove_channels = probe0_highpass.remove_channels(bad_channel_ids)
print('probe0_bad_channel_ids',bad_channel_ids)
#phase shift correction - equivalent to T-SHIFT in catGT
probe0_phase_shift = si.phase_shift(probe0_remove_channels)
probe0_common_reference = si.common_reference(probe0_phase_shift,operator='median',reference='global')
probe0_preprocessed = probe0_common_reference
probe0_cat = si.concatenate_recordings([probe0_preprocessed])
print('probe0_preprocessed',probe0_preprocessed)
print('probe0 concatenated',probe0_cat)

probe1_highpass = si.highpass_filter(probe1_raw,freq_min=300.)
bad_channel_ids, channel_labels = si.detect_bad_channels(probe1_highpass)
probe1_remove_channels = probe1_highpass.remove_channels(bad_channel_ids)
print('probe1_bad_channel_ids',bad_channel_ids)
probe1_phase_shift = si.phase_shift(probe1_remove_channels)
probe1_common_reference = si.common_reference(probe1_phase_shift,operator='median',reference='global')
probe1_preprocessed = probe1_common_reference
probe1_cat = si.concatenate_recordings([probe1_preprocessed])
print('probe1_preprocessed',probe1_preprocessed)
print('probe1 concatenated',probe1_cat)

'''Motion Drift Correction'''
#motion correction if needed
#this is nonrigid correction - need to do parallel computing to speed up
#assign parallel processing as job_kwargs
job_kwargs = dict(n_jobs=20, chunk_duration='1s', progress_bar=True)
probe0_nonrigid_accurate = si.correct_motion(recording=probe0_cat, preset="nonrigid_accurate",**job_kwargs)
probe1_nonrigid_accurate = si.correct_motion(recording=probe1_cat, preset="nonrigid_accurate",**job_kwargs)

print('Start to motion correction finished:')
print(datetime.now() - startTime)
#kilosort like to mimic kilosort - no need if you are just running kilosort
#probe0_kilosort_like = correct_motion(recording=probe0_cat, preset="kilosort_like")
#probe1_kilosort_like = correct_motion(recording=probe1_cat, preset="kilosort_like")

'''save preprocessed bin file before sorting'''

job_kwargs = dict(n_jobs=20, chunk_duration='1s', progress_bar=True)
#after saving, sorters will read this preprocessed binary file instead
probe0_preprocessed_corrected = probe0_nonrigid_accurate.save(folder=dst_folder+'/probe0_preprocessed', format='binary', **job_kwargs)
probe1_preprocessed_corrected = probe1_nonrigid_accurate.save(folder=dst_folder+'/probe1_preprocessed', format='binary', **job_kwargs)
print('Start to prepocessed files saved:')
print(datetime.now() - startTime)
#probe0_preprocessed_corrected = si.load_extractor(dst_folder+'/probe0_preprocessed')
#probe1_preprocessed_corrected = si.load_extractor(dst_folder+'/probe1_preprocessed')
''' prepare sorters - currently using the default parameters and motion correction is turned off as it was corrected already above
    you can check if the parameters using:
    params = get_default_sorter_params('kilosort3')
print("Parameters:\n", params)

desc = get_sorter_params_description('kilosort3')
print("Descriptions:\n", desc)

Beware that moutainsort5 is commented out as the sorter somehow stops midway with no clue - currently raising this issue on their github page
'''
probe0_sorting_ks2_5 = si.run_sorter(sorter_name= 'kilosort2_5',recording=probe0_preprocessed_corrected,output_folder=dst_folder+'/probe0/sorters/kilosort2_5/',docker_image="spikeinterface/kilosort2_5-compiled-base:latest",do_correction=False)
probe1_sorting_ks2_5 = si.run_sorter(sorter_name= 'kilosort2_5',recording=probe1_preprocessed_corrected,output_folder=dst_folder+'/probe1/sorters/kilosort2_5/',docker_image="spikeinterface/kilosort2_5-compiled-base:latest",do_correction=False)
probe0_sorting_ks3 = si.run_sorter(sorter_name= 'kilosort3',recording=probe0_preprocessed_corrected,output_folder=dst_folder+'/probe0/sorters/kilosort3/',docker_image="spikeinterface/kilosort3-compiled-base:latest",do_correction=False)
probe1_sorting_ks3 = si.run_sorter(sorter_name= 'kilosort3',recording=probe1_preprocessed_corrected,output_folder=dst_folder+'/probe1/sorters/kilosort3/',docker_image="spikeinterface/kilosort3-compiled-base:latest",do_correction=False)
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

print('Start to all sorting done:')
print(datetime.now() - startTime)





''' read sorters directly from the output folder - so you dont need to worry if something went wrong and you can't access the temp variables
    This section reads sorter outputs and extract waveforms 
'''
#extract waveforms from sorted data
#job_kwargs = dict(n_jobs=10, chunk_duration='1s', progress_bar=True)#lower n_jobs here - sometimes too many files open for reading the sorter files
#probe0_sorting_ks2_5 = spikeinterface.sorters.read_sorter_folder(dst_folder+'/probe0/sorters/kilosort2_5/', register_recording=True, sorting_info=True, raise_error=True)
probe0_we_ks2_5 = si.extract_waveforms(probe0_preprocessed_corrected, probe0_sorting_ks2_5, folder=dst_folder +'/probe0/waveform/kilosort2_5',
                          sparse=True, max_spikes_per_unit=500, ms_before=1.5,ms_after=2.,
                          **job_kwargs)
del probe0_sorting_ks2_5
#probe0_sorting_ks3 = spikeinterface.sorters.read_sorter_folder(dst_folder+'/probe0/sorters/kilosort3/', register_recording=True, sorting_info=True, raise_error=True)
probe0_we_ks3 = si.extract_waveforms(probe0_preprocessed_corrected, probe0_sorting_ks3, folder=dst_folder +'/probe0/waveform/kilosort3',
                          sparse=True, max_spikes_per_unit=500, ms_before=1.5,ms_after=2.,
                          **job_kwargs)
del probe0_sorting_ks3
#probe1_sorting_ks2_5 = spikeinterface.sorters.read_sorter_folder(dst_folder+'/probe1/sorters/kilosort2_5/', register_recording=True, sorting_info=True, raise_error=True)
probe1_we_ks2_5 = si.extract_waveforms(probe1_preprocessed_corrected, probe1_sorting_ks2_5, folder=dst_folder +'/probe1/waveform/kilosort2_5',
                          sparse=True, max_spikes_per_unit=500, ms_before=1.5,ms_after=2.,
                          **job_kwargs)
del probe1_sorting_ks2_5
#probe1_sorting_ks3 = spikeinterface.sorters.read_sorter_folder(dst_folder+'/probe1/sorters/kilosort3/', register_recording=True, sorting_info=True, raise_error=True)
probe1_we_ks3 = si.extract_waveforms(probe1_preprocessed_corrected, probe1_sorting_ks3, folder=dst_folder +'/probe1/waveform/kilosort3',
                          sparse=True, max_spikes_per_unit=500, ms_before=1.5,ms_after=2.,
                          **job_kwargs)
del probe1_sorting_ks3
''' Compute quality metrics on the extracted waveforms'''
#quality metrics
probe0_ks2_5_metrics = si.compute_quality_metrics(probe0_we_ks2_5, metric_names=['firing_rate', 'presence_ratio', 'snr',
                                                       'isi_violation', 'amplitude_cutoff'])

probe0_ks3_metrics = si.compute_quality_metrics(probe0_we_ks3, metric_names=['firing_rate', 'presence_ratio', 'snr',
                                                       'isi_violation', 'amplitude_cutoff'])
probe1_ks2_5_metrics = si.compute_quality_metrics(probe1_we_ks2_5, metric_names=['firing_rate', 'presence_ratio', 'snr',
                                                       'isi_violation', 'amplitude_cutoff'])
probe1_ks3_metrics = si.compute_quality_metrics(probe1_we_ks3, metric_names=['firing_rate', 'presence_ratio', 'snr',
                                                       'isi_violation', 'amplitude_cutoff'])

'''Apply curation thresholds based on the quality metrics and save the cleaned waveforms'''
#curation - similiar to allen
amplitude_cutoff_thresh = 0.1
isi_violations_ratio_thresh = 1
presence_ratio_thresh = 0.9

our_query = f"(amplitude_cutoff < {amplitude_cutoff_thresh}) & (isi_violations_ratio < {isi_violations_ratio_thresh}) & (presence_ratio > {presence_ratio_thresh})"
probe0_ks2_5_keep_units = probe0_ks2_5_metrics.query(our_query)
probe0_ks2_5_keep_unit_ids = probe0_ks2_5_keep_units.index.values
probe0_ks2_5_we_clean = probe0_we_ks2_5.select_units(probe0_ks2_5_keep_unit_ids, new_folder=dst_folder +'/probe0/waveform_clean/kilosort2_5')

probe0_ks3_keep_units = probe0_ks3_metrics.query(our_query)
probe0_ks3_keep_unit_ids = probe0_ks3_keep_units.index.values
probe0_ks3_we_clean = probe0_we_ks3.select_units(probe0_ks3_keep_unit_ids, new_folder=dst_folder +'/probe0/waveform_clean/kilosort3')

probe1_ks2_5_keep_units = probe1_ks2_5_metrics.query(our_query)
probe1_ks2_5_keep_unit_ids = probe1_ks2_5_keep_units.index.values
probe1_ks2_5_we_clean = probe1_we_ks2_5.select_units(probe1_ks2_5_keep_unit_ids, new_folder=dst_folder +'/probe1/waveform_clean/kilosort2_5')


probe1_ks3_keep_units = probe1_ks3_metrics.query(our_query)
probe1_ks3_keep_unit_ids = probe1_ks3_keep_units.index.values
probe1_ks3_we_clean = probe1_we_ks3.select_units(probe1_ks3_keep_unit_ids, new_folder=dst_folder +'/probe1/waveform_clean/kilosort3')







'''minor corrections to the folder path of files before moving the files to server'''
#process to change all the folder paths in text and .json files on Beast to the server before uploading it to the server
file_list = ["/home/lab/spikeinterface_sorting/temp_data/probe0_preprocessed/provenance.json",
             "/home/lab/spikeinterface_sorting/temp_data/probe1_preprocessed/provenance.json",
             "/home/lab/spikeinterface_sorting/temp_data/probe0/waveform/kilosort2_5/recording.json",
             "/home/lab/spikeinterface_sorting/temp_data/probe0/waveform/kilosort2_5/sorting.json",
             "/home/lab/spikeinterface_sorting/temp_data/probe0/waveform/kilosort3/recording.json",
             "/home/lab/spikeinterface_sorting/temp_data/probe0/waveform/kilosort3/sorting.json",
             "/home/lab/spikeinterface_sorting/temp_data/probe0/waveform_clean/kilosort2_5/recording.json",
             "/home/lab/spikeinterface_sorting/temp_data/probe0/waveform_clean/kilosort2_5/sorting.json",
             "/home/lab/spikeinterface_sorting/temp_data/probe0/waveform_clean/kilosort3/recording.json",
             "/home/lab/spikeinterface_sorting/temp_data/probe0/waveform_clean/kilosort3/sorting.json",
             "/home/lab/spikeinterface_sorting/temp_data/probe0/sorters/kilosort2_5/in_container_sorting/provenance.json",
             "/home/lab/spikeinterface_sorting/temp_data/probe0/sorters/kilosort2_5/in_container_sorting/si_folder.json",
             "/home/lab/spikeinterface_sorting/temp_data/probe0/sorters/kilosort3/in_container_sorting/provenance.json",
             "/home/lab/spikeinterface_sorting/temp_data/probe0/sorters/kilosort3/in_container_sorting/si_folder.json",
             "/home/lab/spikeinterface_sorting/temp_data/probe1/waveform/kilosort2_5/recording.json",
             "/home/lab/spikeinterface_sorting/temp_data/probe1/waveform/kilosort2_5/sorting.json",
             "/home/lab/spikeinterface_sorting/temp_data/probe1/waveform/kilosort3/recording.json",
             "/home/lab/spikeinterface_sorting/temp_data/probe1/waveform/kilosort3/sorting.json",
             "/home/lab/spikeinterface_sorting/temp_data/probe1/waveform_clean/kilosort2_5/recording.json",
             "/home/lab/spikeinterface_sorting/temp_data/probe1/waveform_clean/kilosort2_5/sorting.json",
             "/home/lab/spikeinterface_sorting/temp_data/probe1/waveform_clean/kilosort3/recording.json",
             "/home/lab/spikeinterface_sorting/temp_data/probe1/waveform_clean/kilosort3/sorting.json",
             "/home/lab/spikeinterface_sorting/temp_data/probe1/sorters/kilosort2_5/in_container_sorting/provenance.json",
             "/home/lab/spikeinterface_sorting/temp_data/probe1/sorters/kilosort2_5/in_container_sorting/si_folder.json",
             "/home/lab/spikeinterface_sorting/temp_data/probe1/sorters/kilosort3/in_container_sorting/provenance.json",
             "/home/lab/spikeinterface_sorting/temp_data/probe1/sorters/kilosort3/in_container_sorting/si_folder.json"]

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

beast_folder='/home/lab/spikeinterface_sorting/temp_data/'
for files in file_list:
    
    # open the JSON file and load the data
    with open(files, 'r') as f:
        data = json.load(f)
    
    # replace the text
    data = replace_text(data, beast_folder, ephys_folder)
    
    # write the updated data back to the JSON file
    with open(files, 'w') as f:
        json.dump(data, f, indent=4)


#move spikeinterface folder on Beast to the server

import shutil
import os

folders_to_move = ['probe0',
                   'probe1',
                   'probe0_preprocessed',
                   'probe1_preprocessed']


for folder in folders_to_move:
    # construct the destination path
    destination = os.path.join(ephys_folder, folder)
    # copy the folder to the destination
    shutil.copytree(beast_folder+folder, destination)

#remove all temmp files
shutil.rmtree(beast_folder)

print('All Done! Overall it took:')

print(datetime.now() - startTime)
print('to finish')