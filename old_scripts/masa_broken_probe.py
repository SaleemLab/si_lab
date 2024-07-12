
import os
import subprocess
subprocess.run('ulimit -n 4096',shell=True)
#grab recordings from the server to local machine (Beast)
base_folder = '/mnt/rds01/ibn-vision/DATA/SUBJECTS/'
mouse = 'M23028'
date = '20230703'

job_kwargs = dict(n_jobs=32, chunk_duration='1s', progress_bar=True)
dst_folder = "/home/saleem_lab/spikeinterface_sorting/temp_data/" + date + '/'
# get all the recordings on that day
ephys_folder = base_folder + mouse + '/ephys/' + date +'/'
# allocate destination folder and move the ephys folder on the server to Beast lab user
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
print('copying ephys data from:' + ephys_folder)
g_files = []
# iterate over all directories in source folder

for dirname in os.listdir(ephys_folder):
    # check if '_g' is in the directory name
    #only grab recording folders - there might be some other existing folders for analysis or sorted data
    if '_g' in dirname:
        # construct full directory path
        g_files.append(dirname)
        source = os.path.join(ephys_folder, dirname)
        destination = os.path.join(dst_folder, dirname)
        # copy the directory to the destination folder
        shutil.copytree(source, destination)
print('Start to copying files to Beast:')
print(datetime.now() - startTime)
''' read spikeglx recordings and preprocess them'''
# Define a custom sorting key that extracts the number after 'g'
def sorting_key(s):
    return int(s.split('_g')[-1])
# Sort the list using the custom sorting key
g_files = sorted(g_files, key=sorting_key)
print(g_files) 
stream_names, stream_ids = si.get_neo_streams('spikeglx',dst_folder)
print(stream_names)
print(stream_ids)
#load first probe from beast folder - MEC probe for Diao
probe0_raw = si.read_spikeglx(dst_folder,stream_name='imec0.ap')
print(probe0_raw)
#Load second probe - V1 probe
probe1_raw = si.read_spikeglx(dst_folder,stream_name = 'imec1.ap')
print(probe1_raw)
import itertools
probe0_num_segments = [probe0_raw.get_num_frames(segment_index=i) for i in range(probe0_raw.get_num_segments())]
probe1_num_segments = [probe1_raw.get_num_frames(segment_index=i) for i in range(probe0_raw.get_num_segments())]
probe0_end_sample_frames = list(itertools.accumulate(probe0_num_segments))
probe0_start_sample_frames = [1] + [probe0_end_sample_frames[i] + 1 for i in range(0, len(probe0_num_segments)-1)]
probe1_end_sample_frames = list(itertools.accumulate(probe1_num_segments))
probe1_start_sample_frames = [1] + [probe1_end_sample_frames[i] + 1 for i in range(0, len(probe1_num_segments)-1)]
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
probe0_cat = probe0_cat.frame_slice( start_frame=0, end_frame=(probe0_start_sample_frames[1]-1))
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

probe0_nonrigid_accurate = si.correct_motion(recording=probe0_cat, preset="nonrigid_accurate",**job_kwargs)
probe1_nonrigid_accurate = si.correct_motion(recording=probe1_cat, preset="nonrigid_accurate",**job_kwargs)
print('Start to motion correction finished:')
print(datetime.now() - startTime)
#kilosort like to mimic kilosort - no need if you are just running kilosort
#probe0_kilosort_like = correct_motion(recording=probe0_cat, preset="kilosort_like")
#probe1_kilosort_like = correct_motion(recording=probe1_cat, preset="kilosort_like")
'''save preprocessed bin file before sorting'''
#after saving, sorters will read this preprocessed binary file instead
probe0_preprocessed_corrected = probe0_nonrigid_accurate.save(folder=dst_folder+'probe0_preprocessed', format='binary', **job_kwargs)
probe1_preprocessed_corrected = probe1_nonrigid_accurate.save(folder=dst_folder+'probe1_preprocessed', format='binary', **job_kwargs)
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
probe0_sorting_ks2_5 = si.run_sorter(sorter_name= 'kilosort2_5',recording=probe0_preprocessed_corrected,output_folder=dst_folder+'probe0/sorters/kilosort2_5/',docker_image="spikeinterface/kilosort2_5-compiled-base:latest",do_correction=False)
probe1_sorting_ks2_5 = si.run_sorter(sorter_name= 'kilosort2_5',recording=probe1_preprocessed_corrected,output_folder=dst_folder+'probe1/sorters/kilosort2_5/',docker_image="spikeinterface/kilosort2_5-compiled-base:latest",do_correction=False)
probe0_sorting_ks3 = si.run_sorter(sorter_name= 'kilosort3',recording=probe0_preprocessed_corrected,output_folder=dst_folder+'probe0/sorters/kilosort3/',docker_image="spikeinterface/kilosort3-compiled-base:latest",do_correction=False)
probe1_sorting_ks3 = si.run_sorter(sorter_name= 'kilosort3',recording=probe1_preprocessed_corrected,output_folder=dst_folder+'probe1/sorters/kilosort3/',docker_image="spikeinterface/kilosort3-compiled-base:latest",do_correction=False)
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
probe0_sorting_ks2_5 = si.remove_duplicated_spikes(sorting = probe0_sorting_ks2_5, censored_period_ms=0.3,method='keep_first')
probe0_sorting_ks3 = si.remove_duplicated_spikes(sorting = probe0_sorting_ks3, censored_period_ms=0.3,method='keep_first')
probe1_sorting_ks2_5 = si.remove_duplicated_spikes(sorting = probe1_sorting_ks2_5, censored_period_ms=0.3,method='keep_first')
probe1_sorting_ks3 = si.remove_duplicated_spikes(sorting = probe1_sorting_ks3, censored_period_ms=0.3,method='keep_first')
print(probe0_sorting_ks2_5)
print(probe0_sorting_ks3)
print(probe1_sorting_ks2_5)
print(probe1_sorting_ks3)
print('Start to all sorting done:')
print(datetime.now() - startTime)
import pandas as pd
probe0_segment_frames = pd.DataFrame({'segment_info':g_files,'segment start frame': probe0_start_sample_frames, 'segment end frame': probe0_end_sample_frames})
probe0_segment_frames.to_csv(dst_folder+'probe0/sorters/segment_frames.csv', index=False)
probe1_segment_frames = pd.DataFrame({'segment_info':g_files,'segment start frame': probe1_start_sample_frames, 'segment end frame': probe1_end_sample_frames})
probe1_segment_frames.to_csv(dst_folder+'probe1/sorters/segment_frames.csv', index=False)
''' read sorters directly from the output folder - so you dont need to worry if something went wrong and you can't access the temp variables
    This section reads sorter outputs and extract waveforms 
'''
#extract waveforms from sorted data

#probe0_sorting_ks2_5 = spikeinterface.sorters.read_sorter_folder(dst_folder+'/probe0/sorters/kilosort2_5/', register_recording=True, sorting_info=True, raise_error=True)
probe0_we_ks2_5 = si.extract_waveforms(probe0_preprocessed_corrected, probe0_sorting_ks2_5, folder=dst_folder +'probe0/waveform/kilosort2_5',
                        sparse=True, max_spikes_per_unit=500, ms_before=1.5,ms_after=2.,
                        **job_kwargs)
del probe0_sorting_ks2_5
#probe0_sorting_ks3 = spikeinterface.sorters.read_sorter_folder(dst_folder+'/probe0/sorters/kilosort3/', register_recording=True, sorting_info=True, raise_error=True)
probe0_we_ks3 = si.extract_waveforms(probe0_preprocessed_corrected, probe0_sorting_ks3, folder=dst_folder +'probe0/waveform/kilosort3',
                        sparse=True, max_spikes_per_unit=500, ms_before=1.5,ms_after=2.,
                        **job_kwargs)
del probe0_sorting_ks3
#probe1_sorting_ks2_5 = spikeinterface.sorters.read_sorter_folder(dst_folder+'/probe1/sorters/kilosort2_5/', register_recording=True, sorting_info=True, raise_error=True)
probe1_we_ks2_5 = si.extract_waveforms(probe1_preprocessed_corrected, probe1_sorting_ks2_5, folder=dst_folder +'probe1/waveform/kilosort2_5',
                        sparse=True, max_spikes_per_unit=500, ms_before=1.5,ms_after=2.,
                        **job_kwargs)
del probe1_sorting_ks2_5
#probe1_sorting_ks3 = spikeinterface.sorters.read_sorter_folder(dst_folder+'/probe1/sorters/kilosort3/', register_recording=True, sorting_info=True, raise_error=True)
probe1_we_ks3 = si.extract_waveforms(probe1_preprocessed_corrected, probe1_sorting_ks3, folder=dst_folder +'probe1/waveform/kilosort3',
                        sparse=True, max_spikes_per_unit=500, ms_before=1.5,ms_after=2.,
                        **job_kwargs)
del probe1_sorting_ks3
''' Compute quality metrics on the extracted waveforms'''
template_metric_probe0_ks2_5 = si.compute_template_metrics(probe0_we_ks2_5)
template_metric_probe0_ks3 = si.compute_template_metrics(probe0_we_ks3)
template_metric_probe1_ks2_5 = si.compute_template_metrics(probe1_we_ks2_5)
template_metric_probe1_ks3 = si.compute_template_metrics(probe1_we_ks3)
noise_levels_probe0_ks2_5 = si.compute_noise_levels(probe0_we_ks2_5)
noise_levels_probe0_ks3 = si.compute_noise_levels(probe0_we_ks3)
noise_levels_probe1_ks2_5 = si.compute_noise_levels(probe1_we_ks2_5)
noise_levels_probe1_ks3 = si.compute_noise_levels(probe1_we_ks3)
PCA_probe0_ks2_5 = si.compute_principal_components(probe0_we_ks2_5,**job_kwargs)
PCA_probe0_ks3 = si.compute_principal_components(probe0_we_ks3,**job_kwargs)
PCA_probe1_ks2_5 = si.compute_principal_components(probe1_we_ks2_5,**job_kwargs)
PCA_probe1_ks3 = si.compute_principal_components(probe1_we_ks3,**job_kwargs)
template_similarity_probe0_ks2_5 = si.compute_template_similarity(probe0_we_ks2_5)
template_similarity_probe0_ks3 = si.compute_template_similarity(probe0_we_ks3)
template_similarity_probe1_ks2_5 = si.compute_template_similarity(probe1_we_ks2_5)
template_similarity_probe1_ks3 = si.compute_template_similarity(probe1_we_ks3)
correlograms_probe0_ks2_5 = si.compute_correlograms(probe0_we_ks2_5)
correlograms_probe0_ks3 = si.compute_correlograms(probe0_we_ks3)
correlograms_probe1_ks2_5 = si.compute_correlograms(probe1_we_ks2_5)
correlograms_probe1_ks3 = si.compute_correlograms(probe1_we_ks3)
amplitudes_probe0_ks2_5 = si.compute_spike_amplitudes(probe0_we_ks2_5,**job_kwargs)
amplitudes_probe0_ks3 = si.compute_spike_amplitudes(probe0_we_ks3,**job_kwargs)
amplitudes_probe1_ks2_5 = si.compute_spike_amplitudes(probe1_we_ks2_5,**job_kwargs)
amplitudes_probe1_ks3 = si.compute_spike_amplitudes(probe1_we_ks3,**job_kwargs)
isi_histograms_probe0_ks2_5 = si.compute_isi_histograms(probe0_we_ks2_5)
isi_histograms_probe0_ks3 = si.compute_isi_histograms(probe0_we_ks3)
isi_histograms_probe1_ks2_5 = si.compute_isi_histograms(probe1_we_ks2_5)
isi_histograms_probe1_ks3 = si.compute_isi_histograms(probe1_we_ks3)
qm_list = si.get_quality_metric_list()
print('The following quality metrics are computed:')
print(qm_list)
probe0_ks2_5_metrics = si.compute_quality_metrics(probe0_we_ks2_5, metric_names=qm_list,**job_kwargs)
probe0_ks3_metrics = si.compute_quality_metrics(probe0_we_ks3, metric_names=qm_list,**job_kwargs)
probe1_ks2_5_metrics = si.compute_quality_metrics(probe1_we_ks2_5, metric_names=qm_list,**job_kwargs)
probe1_ks3_metrics = si.compute_quality_metrics(probe1_we_ks3, metric_names=qm_list,**job_kwargs)
'''minor corrections to the folder path of files before moving the files to server'''
#process to change all the folder paths in text and .json files on Beast to the server before uploading it to the server
file_list = [dst_folder + "probe0_preprocessed/provenance.json",
            dst_folder + "probe1_preprocessed/provenance.json",
            dst_folder + "probe0/waveform/kilosort2_5/recording.json",
            dst_folder + "probe0/waveform/kilosort2_5/sorting.json",
            dst_folder + "probe0/waveform/kilosort3/recording.json",
            dst_folder + "probe0/waveform/kilosort3/sorting.json",
            dst_folder + "probe0/sorters/kilosort2_5/in_container_sorting/provenance.json",
            dst_folder + "probe0/sorters/kilosort2_5/in_container_sorting/si_folder.json",
            dst_folder + "probe0/sorters/kilosort3/in_container_sorting/provenance.json",
            dst_folder + "probe0/sorters/kilosort3/in_container_sorting/si_folder.json",
            dst_folder + "probe1/waveform/kilosort2_5/recording.json",
            dst_folder + "probe1/waveform/kilosort2_5/sorting.json",
            dst_folder + "probe1/waveform/kilosort3/recording.json",
            dst_folder + "probe1/waveform/kilosort3/sorting.json",
            dst_folder + "probe1/sorters/kilosort2_5/in_container_sorting/provenance.json",
            dst_folder + "probe1/sorters/kilosort2_5/in_container_sorting/si_folder.json",
            dst_folder + "probe1/sorters/kilosort3/in_container_sorting/provenance.json",
            dst_folder + "probe1/sorters/kilosort3/in_container_sorting/si_folder.json"]
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
#delete temp_wh.dat files
dat_files = [ dst_folder + "probe0/sorters/kilosort2_5/sorter_output/temp_wh.dat",
            dst_folder + "probe0/sorters/kilosort3/sorter_output/temp_wh.dat",
            dst_folder + "probe1/sorters/kilosort2_5/sorter_output/temp_wh.dat",
            dst_folder + "probe1/sorters/kilosort3/sorter_output/temp_wh.dat"]
for files in dat_files:
    os.remove(files)
#move spikeinterface folder on Beast to the server
import shutil
import os
folders_to_move = ['probe0',
                 'probe1',
                 'probe0_preprocessed',
                 'probe1_preprocessed']


for folder in folders_to_move:
     # construct the destination path
#    destination = os.path.join(ephys_folder, folder)
 #   # copy the folder to the destination
  #  shutil.copytree(dst_folder+folder, destination)

#remove all temmp files
#shutil.rmtree(dst_folder)
print('All Done! Overall it took:')
print(datetime.now() - startTime)
print('to finish! Please move the files to the server as soon as you have time!')