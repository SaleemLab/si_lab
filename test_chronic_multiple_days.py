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

base_folder = '/mnt/rds01/ibn-vision/DATA/SUBJECTS/'
mouse = 'M23087'
dates = ['20231207','20231212']
dst_folder = "/home/saleem_lab/spikeinterface_sorting/temp_data/"
job_kwargs = dict(n_jobs=32, chunk_duration='1s', progress_bar=True)
# get all the recordings on that day
ephys_folder = []
ephys_folder.append(base_folder + mouse + '/ephys/' + dates[0] +'/')
ephys_folder.append(base_folder + mouse + '/ephys/' + dates[1] +'/')
# allocate destination folder and move the ephys folder on the server to Beast lab user

startTime = datetime.now()
print('Start Time:' + startTime.strftime("%m/%d/%Y, %H:%M:%S"))
''' this section defines the animal and dates and fetch the recordings from the server to Beast'''

g_files = []
# iterate over all directories in source folder
for d in range(0,len(ephys_folder)):
    print('copying ephys data from:' + ephys_folder[d])
    for dirname in os.listdir(ephys_folder[d]):
        
        # check if '_g' is in the directory name
        #only grab recording folders - there might be some other existing folders for analysis or sorted data
        if '_g' in dirname:
            # construct full directory path
            g_files.append(dirname)
            source = os.path.join(ephys_folder[d], dirname)
            destination = os.path.join(dst_folder+dates[d], dirname)
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
#load first day recordings
day0_raw = si.read_spikeglx(dst_folder+dates[0],stream_name='imec0.ap')
print(day0_raw)
#Load second day recordings
day1_raw = si.read_spikeglx(dst_folder+dates[1],stream_name = 'imec0.ap')
print(day1_raw)
import itertools
day0_num_segments = [day0_raw.get_num_frames(segment_index=i) for i in range(day0_raw.get_num_segments())]
day1_num_segments = [day1_raw.get_num_frames(segment_index=i) for i in range(day1_raw.get_num_segments())]
#join the recording segment numbers
num_segments = day0_num_segments + day1_num_segments
end_sample_frames = list(itertools.accumulate(num_segments))
start_sample_frames = [1] + [end_sample_frames[i] + 1 for i in range(0, len(num_segments)-1)]

#several preprocessing steps and concatenation of the recordings
#highpass filter - threhsold at 300Hz
day0_highpass = si.highpass_filter(day0_raw,freq_min=300.)
#detect bad channels
bad_channel_ids, channel_labels = si.detect_bad_channels(day0_highpass)
#remove bad channels if wanted
day0_remove_channels = day0_highpass.remove_channels(bad_channel_ids)
print('day0_bad_channel_ids',bad_channel_ids)
#phase shift correction - equivalent to T-SHIFT in catGT
day0_phase_shift = si.phase_shift(day0_remove_channels)
day0_common_reference = si.common_reference(day0_phase_shift,operator='median',reference='global')
day0_preprocessed = day0_common_reference
day0_cat = si.concatenate_recordings([day0_preprocessed])
print('day0_preprocessed',day0_preprocessed)
print('day0 concatenated',day0_cat)
day1_highpass = si.highpass_filter(day1_raw,freq_min=300.)
bad_channel_ids, channel_labels = si.detect_bad_channels(day1_highpass)
day1_remove_channels = day1_highpass.remove_channels(bad_channel_ids)
print('day1_bad_channel_ids',bad_channel_ids)
day1_phase_shift = si.phase_shift(day1_remove_channels)
day1_common_reference = si.common_reference(day1_phase_shift,operator='median',reference='global')
day1_preprocessed = day1_common_reference
day1_cat = si.concatenate_recordings([day1_preprocessed])
print('day1_preprocessed',day1_preprocessed)
print('day1 concatenated',day1_cat)

cat_all_days = si.concatenate_recordings([day0_cat,day1_cat])
'''Motion Drift Correction'''
#motion correction if needed
#this is nonrigid correction - need to do parallel computing to speed up
#assign parallel processing as job_kwargs

all_cat_nonrigid_accurate = si.correct_motion(recording=cat_all_days, preset="nonrigid_accurate",**job_kwargs)

print('Start to motion correction finished:')
print(datetime.now() - startTime)
#kilosort like to mimic kilosort - no need if you are just running kilosort
#probe0_kilosort_like = correct_motion(recording=probe0_cat, preset="kilosort_like")
#probe1_kilosort_like = correct_motion(recording=probe1_cat, preset="kilosort_like")
'''save preprocessed bin file before sorting'''
#after saving, sorters will read this preprocessed binary file instead
all_preprocessed_corrected = all_cat_nonrigid_accurate.save(folder=dst_folder+'all_preprocessed', format='binary', **job_kwargs)
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
all_sorting_ks2_5 = si.run_sorter(sorter_name= 'kilosort2_5',recording=all_preprocessed_corrected,output_folder=dst_folder+'all/sorters/kilosort2_5/',docker_image="spikeinterface/kilosort2_5-compiled-base:latest",do_correction=False)
all_sorting_ks3 = si.run_sorter(sorter_name= 'kilosort3',recording=all_preprocessed_corrected,output_folder=dst_folder+'all/sorters/kilosort3/',docker_image="spikeinterface/kilosort3-compiled-base:latest",do_correction=False)
#sortings = si.run_sorter_jobs(job_list = job_list,engine = 'joblib',engine_kwargs = {'n_jobs': 2})
#remove duplicates
all_sorting_ks2_5 = si.remove_duplicated_spikes(sorting = all_sorting_ks2_5, censored_period_ms=0.3,method='keep_first')
all_sorting_ks3 = si.remove_duplicated_spikes(sorting = all_sorting_ks3, censored_period_ms=0.3,method='keep_first')
print(all_sorting_ks2_5)
print(all_sorting_ks3)
p
print('Start to all sorting done:')
print(datetime.now() - startTime)
import pandas as pd
segment_frames = pd.DataFrame({'segment_info':g_files,'segment start frame': start_sample_frames, 'segment end frame': end_sample_frames})
segment_frames.to_csv(dst_folder+'all/sorters/segment_frames.csv', index=False)

''' read sorters directly from the output folder - so you dont need to worry if something went wrong and you can't access the temp variables
    This section reads sorter outputs and extract waveforms 
'''
#extract waveforms from sorted data

#probe0_sorting_ks2_5 = spikeinterface.sorters.read_sorter_folder(dst_folder+'/probe0/sorters/kilosort2_5/', register_recording=True, sorting_info=True, raise_error=True)
all_we_ks2_5 = si.extract_waveforms(all_preprocessed_corrected, all_sorting_ks2_5, folder=dst_folder +'all/waveform/kilosort2_5',
                        sparse=True, max_spikes_per_unit=500, ms_before=1.5,ms_after=2.,
                        **job_kwargs)

#probe0_sorting_ks3 = spikeinterface.sorters.read_sorter_folder(dst_folder+'/probe0/sorters/kilosort3/', register_recording=True, sorting_info=True, raise_error=True)
all_we_ks3 = si.extract_waveforms(all_preprocessed_corrected, all_sorting_ks3, folder=dst_folder +'all/waveform/kilosort3',
                        sparse=True, max_spikes_per_unit=500, ms_before=1.5,ms_after=2.,
                        **job_kwargs)

''' Compute quality metrics on the extracted waveforms'''
template_metric_all_ks2_5 = si.compute_template_metrics(all_we_ks2_5)
template_metric_all_ks3 = si.compute_template_metrics(all_we_ks3)

noise_levels_all_ks2_5 = si.compute_noise_levels(all_we_ks2_5)
noise_levels_all_ks3 = si.compute_noise_levels(all_we_ks3)

PCA_all_ks2_5 = si.compute_principal_components(all_we_ks2_5,**job_kwargs)
PCA_all_ks3 = si.compute_principal_components(all_we_ks3,**job_kwargs)

template_similarity_all_ks2_5 = si.compute_template_similarity(all_we_ks2_5)
template_similarity_all_ks3 = si.compute_template_similarity(all_we_ks3)

correlograms_all_ks2_5 = si.compute_correlograms(all_we_ks2_5)
correlograms_all_ks3 = si.compute_correlograms(all_we_ks3)

amplitudes_all_ks2_5 = si.compute_spike_amplitudes(all_we_ks2_5,**job_kwargs)
amplitudes_all_ks3 = si.compute_spike_amplitudes(all_we_ks3,**job_kwargs)

isi_histograms_all_ks2_5 = si.compute_isi_histograms(all_we_ks2_5)
isi_histograms_all_ks3 = si.compute_isi_histograms(all_we_ks3)

qm_list = si.get_quality_metric_list()
print('The following quality metrics are computed:')
print(qm_list)
all_ks2_5_metrics = si.compute_quality_metrics(all_we_ks2_5, metric_names=qm_list,**job_kwargs)
all_ks3_metrics = si.compute_quality_metrics(all_we_ks3, metric_names=qm_list,**job_kwargs)

'''minor corrections to the folder path of files before moving the files to server'''
#process to change all the folder paths in text and .json files on Beast to the server before uploading it to the server
file_list = [dst_folder + "all_preprocessed/provenance.json",
            dst_folder + "all/waveform/kilosort2_5/recording.json",
            dst_folder + "all/waveform/kilosort2_5/sorting.json",
            dst_folder + "all/waveform/kilosort3/recording.json",
            dst_folder + "all/waveform/kilosort3/sorting.json",
            dst_folder + "all/sorters/kilosort2_5/in_container_sorting/provenance.json",
            dst_folder + "all/sorters/kilosort2_5/in_container_sorting/si_folder.json",
            dst_folder + "all/sorters/kilosort3/in_container_sorting/provenance.json",
            dst_folder + "all/sorters/kilosort3/in_container_sorting/si_folder.json"]
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
    data = replace_text(data, dst_folder[:-1], ephys_folder[0][:-1]+'_'+dates[1])
    
    # write the updated data back to the JSON file
    with open(files, 'w') as f:
        json.dump(data, f, indent=4)
#delete temp_wh.dat files
dat_files = [ dst_folder + "all/sorters/kilosort2_5/sorter_output/temp_wh.dat",
            dst_folder + "all/sorters/kilosort3/sorter_output/temp_wh.dat"]
for files in dat_files:
    os.remove(files)
#move spikeinterface folder on Beast to the server
import shutil
import os
folders_to_move = ['all',
                'all_preprocessed']
for folder in folders_to_move:
    # construct the destination path
    destination = os.path.join(ephys_folder[0][:-1]+'_'+dates[1], folder)
    # copy the folder to the destination
    shutil.copytree(dst_folder+folder, destination)
#remove all temmp files
shutil.rmtree(dst_folder)
print('All Done! Overall it took:')
print(datetime.now() - startTime)
print('to finish')
