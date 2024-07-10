
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
from datetime import datetime
import itertools

import scipy.io as sio
startTime = datetime.now()
print('Start Time:' + startTime.strftime("%m/%d/%Y, %H:%M:%S"))
''' this section defines the animal and dates and fetch the recordings from the server to Beast'''
mouse = 'M24016'
session = '20240706'
dates = ['20240706/20240608_0','']
g_files_to_ignore = ['tcat','6','7','8','9']

save_date = '20240706'
base_folder = 'Z:/ibn-vision/DATA/SUBJECTS/'
save_base_folder = base_folder
save_folder = save_base_folder + mouse + '/ephys/' + session + '/'
# get all the recordings on that day
probe0_start_sample_fames = []
probe1_start_sample_frames = []
probe0_end_sample_frames = []
probe1_end_sample_frames = []
import os
import subprocess
subprocess.run('ulimit -n 4096',shell=True)
def sorting_key(s):
    return int(s.split('_g')[-1])

#grab recordings from the server to local machine (Beast)


job_kwargs = dict(n_jobs=8, chunk_duration='1s', progress_bar=True)


g_files_all = []
# iterate over all directories in source folder
date_count = 0
for date in dates:
    date_count = date_count + 1
    ephys_folder = base_folder + mouse + '/ephys/' + date +'/'
    dst_folder =  save_base_folder + date + '/'
    ephys_folder = base_folder + mouse + '/ephys/' + date +'/'
    g_files = []
    # print('copying ephys data from:' + ephys_folder)
    for dirname in os.listdir(ephys_folder):
        # ignore some folders or files includiing tcat and other G files (such as checkerboard recordings)
        if any(ignore_str in dirname for ignore_str in g_files_to_ignore):
            continue
        # check if '_g' is in the directory name
        # only grab recording folders - there might be some other existing folders for analysis or sorted data
        if '_g' in dirname:
            # construct full directory path
            g_files.append(dirname)
            source = os.path.join(ephys_folder, dirname)
            destination = os.path.join(dst_folder, dirname)
            # copy the directory to the destination folder
            # shutil.copytree(source, destination)
    print('Start to copying files to Beast:')
    print(datetime.now() - startTime)
    ''' read spikeglx recordings and preprocess them'''
    # Define a custom sorting key that extracts the number after 'g'

    # Sort the list using the custom sorting key
    g_files = sorted(g_files, key=sorting_key)
    g_files_all = g_files_all + g_files
    print(g_files)
    print('all g files:', g_files_all)
    stream_names, stream_ids = si.get_neo_streams('spikeglx', dst_folder)
    print(stream_names)
    print(stream_ids)
    # load first probe from beast folder - MEC probe for Diao
    probe0_raw = si.read_spikeglx(dst_folder, stream_name='imec0.ap')
    print(probe0_raw)
    # Load second probe - V1 probe
    probe1_raw = si.read_spikeglx(dst_folder, stream_name='imec1.ap')
    print(probe1_raw)

    probe0_num_segments = [probe0_raw.get_num_frames(segment_index=i) for i in range(probe0_raw.get_num_segments())]
    probe1_num_segments = [probe1_raw.get_num_frames(segment_index=i) for i in range(probe0_raw.get_num_segments())]

    probe0_end_sample_frames_tmp = list(itertools.accumulate(probe0_num_segments))
    if date_count == 1:
        probe0_start_sample_frames = [1] + [probe0_end_sample_frames_tmp[i] + 1 for i in range(0, len(probe0_num_segments) - 1)]
        probe0_end_sample_frames = probe0_end_sample_frames + probe0_end_sample_frames_tmp
    else:
        probe0_start_sample_frames = probe0_start_sample_frames + [probe0_end_sample_frames[-1] + 1] +
    #     # check if '_g' is in the directory name
    #     #only grab recording folders - there might be some other existing folders for analysis or sorted data
        if '_g' in dirname:
    #         # construct full directory path
            g_files.append(dirname)
            source = os.path.join(ephys_folder, dirname)
            destination = os.path.join(dst_folder, dirname)
            # copy the directory to the destination folder
            #shutil.copytree(source, destination)
    print('Start to copying files to Beast:')
    print(datetime.now() - startTime)
    ''' read spikeglx recordings and preprocess them'''
    # Define a custom sorting key that extracts the number after 'g'

    # Sort the list using the custom sorting key
    g_files = sorted(g_files, key=sorting_key)
    g_files_all = g_files_all + g_files
    print(g_files)
    print('all g files:',g_files_all) 
    stream_names, stream_ids = si.get_neo_streams('spikeglx',dst_folder)
    print(stream_names)
    print(stream_ids)
    #load first probe from beast folder - MEC probe for Diao
    probe0_raw = si.read_spikeglx(dst_folder,stream_name='imec0.ap')
    print(probe0_raw)
    #Load second probe - V1 probe
    probe1_raw = si.read_spikeglx(dst_folder,stream_name = 'imec1.ap')
    print(probe1_raw)

    probe0_num_segments = [probe0_raw.get_num_frames(segment_index=i) for i in range(probe0_raw.get_num_segments())]
    probe1_num_segments = [probe1_raw.get_num_frames(segment_index=i) for i in range(probe0_raw.get_num_segments())]
    
    probe0_end_sample_frames_tmp = list(itertools.accumulate(probe0_num_segments))
    if date_count == 1:
        probe0_start_sample_frames = [1] + [probe0_end_sample_frames_tmp[i] + 1 for i in range(0, len(probe0_num_segments)-1)]
        probe0_end_sample_frames = probe0_end_sample_frames + probe0_end_sample_frames_tmp
    else:
        probe0_start_sample_frames = probe0_start_sample_frames + [probe0_end_sample_frames[-1]+1] + \
        [probe0_end_sample_frames[-1]+probe0_end_sample_frames_tmp[i] + 1 for i in range(0, len(probe0_num_segments)-1)]
        probe0_end_sample_frames = probe0_end_sample_frames + [probe0_end_sample_frames_tmp[i] + probe0_end_sample_frames[-1] for i in range(0, len(probe0_num_segments))]

        
    probe1_end_sample_frames_tmp = list(itertools.accumulate(probe1_num_segments))
    if date_count == 1:
        probe1_start_sample_frames = [1] + [probe1_end_sample_frames_tmp[i] + 1 for i in range(0, len(probe1_num_segments)-1)]
        probe1_end_sample_frames = probe1_end_sample_frames + probe1_end_sample_frames_tmp
    else:
        probe1_start_sample_frames = probe1_start_sample_frames + [probe1_end_sample_frames[-1]+1] + \
        [probe1_end_sample_frames[-1]+probe1_end_sample_frames_tmp[i] + 1 for i in range(0, len(probe1_num_segments)-1)]
        probe1_end_sample_frames = probe1_end_sample_frames + [probe1_end_sample_frames_tmp[i] + probe1_end_sample_frames[-1] for i in range(0, len(probe1_num_segments))]

    #several preprocessing steps and concatenation of the recordings
    #highpass filter - threhsold at 300Hz
    probe0_highpass = si.highpass_filter(probe0_raw,freq_min=300.)
    #detect bad channels

    #phase shift correction - equivalent to T-SHIFT in catGT
    probe0_phase_shift = si.phase_shift(probe0_highpass)
    probe0_common_reference = si.common_reference(probe0_phase_shift,operator='median',reference='global')
    probe0_preprocessed = probe0_common_reference
    probe0_cat = si.concatenate_recordings([probe0_preprocessed])
    print('probe0_preprocessed',probe0_preprocessed)
    print('probe0 concatenated',probe0_cat)

    probe1_highpass = si.highpass_filter(probe1_raw,freq_min=300.)
    probe1_phase_shift = si.phase_shift(probe1_highpass)
    probe1_common_reference = si.common_reference(probe1_phase_shift,operator='median',reference='global')
    probe1_preprocessed = probe1_common_reference
    probe1_cat = si.concatenate_recordings([probe1_preprocessed])
    print('probe1_preprocessed',probe1_preprocessed)
    print('probe1 concatenated',probe1_cat)
    
    if date_count == 1:
        probe0_cat_all = probe0_cat
        probe1_cat_all = probe1_cat
    else:
        probe0_cat_all = si.concatenate_recordings([probe0_cat_all,probe0_cat])
        probe1_cat_all = si.concatenate_recordings([probe1_cat_all,probe1_cat])
bad_channel_ids, channel_labels = si.detect_bad_channels(probe0_cat_all)
probe0_cat_all = probe0_cat_all.remove_channels(bad_channel_ids)
print('probe0_bad_channel_ids',bad_channel_ids)
bad_channel_ids, channel_labels = si.detect_bad_channels(probe1_cat_all)
probe1_cat_all = probe1_cat_all.remove_channels(bad_channel_ids)
print('probe1_bad_channel_ids',bad_channel_ids)
'''Motion Drift Correction'''
#motion correction if needed
#this is nonrigid correction - need to do parallel computing to speed up
#assign parallel processing as job_kwargs

probe0_nonrigid_accurate = si.correct_motion(recording=probe0_cat_all, preset="nonrigid_accurate",**job_kwargs)
probe1_nonrigid_accurate = si.correct_motion(recording=probe1_cat_all, preset="nonrigid_accurate",**job_kwargs)

print('Start to motion correction finished:')
print(datetime.now() - startTime)
#kilosort like to mimic kilosort - no need if you are just running kilosort
# probe0_kilosort_like = correct_motion(recording=probe0_cat, preset="kilosort_like")
# probe1_kilosort_like = correct_motion(recording=probe1_cat, preset="kilosort_like")

'''save preprocessed bin file before sorting'''

import pandas as pd
probe0_segment_frames = pd.DataFrame({'segment_info':g_files_all,'segment start frame': probe0_start_sample_frames, 'segment end frame': probe0_end_sample_frames})
probe0_segment_frames.to_csv(save_folder+'probe0/sorters/segment_frames.csv', index=False)
probe1_segment_frames = pd.DataFrame({'segment_info':g_files_all,'segment start frame': probe1_start_sample_frames, 'segment end frame': probe1_end_sample_frames})
probe1_segment_frames.to_csv(save_folder+'probe1/sorters/segment_frames.csv', index=False)

#after saving, sorters will read this preprocessed binary file instead
probe0_preprocessed_corrected = probe0_nonrigid_accurate.save(folder=save_folder+'probe0_preprocessed', format='binary', **job_kwargs)
probe1_preprocessed_corrected = probe1_nonrigid_accurate.save(folder=save_folder+'probe1_preprocessed', format='binary', **job_kwargs)
print('Start to prepocessed files saved:')
print(datetime.now() - startTime)
print('Preprocessing finished')