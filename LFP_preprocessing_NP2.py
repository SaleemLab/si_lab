
# get all the recordings on that day
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

# import docker
from datetime import datetime
import itertools
# import matlab.engine
import scipy.io as sio
startTime = datetime.now()
print('Start Time:' + startTime.strftime("%m/%d/%Y, %H:%M:%S"))
''' this section defines the animal and dates and fetch the recordings from the server to Beast'''
mouse = 'M24017'
dates = ['20240603/20240603_0','20240605/20240605']
save_date = '20240603'

base_folder = 'Z:/ibn-vision/DATA/SUBJECTS/'
save_folder = '/home/lab/spikeinterface_sorting/temp_data/'+save_date+'/'
# get all the recordings on that day
probe0_start_sample_frames = []
probe1_start_sample_frames = []
probe0_end_sample_frames = []
probe1_end_sample_frames = []
import os
#import subprocess
#subprocess.run('ulimit -n 4096',shell=True)
def sorting_key(s):
    return int(s.split('_g')[-1])

job_kwargs = dict(n_jobs=10, chunk_duration='1s', progress_bar=True)


g_files_all = []
# iterate over all directories in source folder
date_count = 0
for date in dates:
    date_count = date_count + 1
    ephys_folder = base_folder + mouse + '/ephys/' + date +'/'
    ephys_folder = base_folder + mouse + '/ephys/' + date +'/'
    g_files = []
    # print('copying ephys data from:' + ephys_folder)
    for dirname in os.listdir(ephys_folder):
    #     # check if '_g' is in the directory name
    #     #only grab recording folders - there might be some other existing folders for analysis or sorted data
        if '_g' in dirname:
    #         # construct full directory path
            g_files.append(dirname)
            #source = os.path.join(ephys_folder, dirname)
            #destination = os.path.join(dst_folder, dirname)
            # copy the directory to the destination folder
            #shutil.copytree(source, destination)
    ''' read spikeglx recordings and preprocess them'''
    # Define a custom sorting key that extracts the number after 'g'

    # Sort the list using the custom sorting key
    g_files = sorted(g_files, key=sorting_key)
    g_files_all = g_files_all + g_files
    print(g_files)
    print('all g files:',g_files_all) 

    for g_file in g_files:
        dst_folder = ephys_folder + g_file + '/'
                                  
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
        #lowpass filter - from 0.5 to 600 Hz
        probe0_lowpass = si.bandpass_filter(probe0_raw,freq_min=0.5, freq_max=600)
        probe0_lowpass = si.resample(probe0_lowpass, 1250)
        #phase shift correction - equivalent to T-SHIFT in catGT
        probe0_phase_shift = si.phase_shift(probe0_lowpass)
        probe0_common_reference = si.common_reference(probe0_phase_shift,operator='median',reference='global')
        probe0_preprocessed = probe0_common_reference
        print('probe0_preprocessed',probe0_preprocessed)


        #lowpass filter - from 0.5 to 600 Hz
        probe1_lowpass = si.bandpass_filter(probe0_raw,freq_min=0.5, freq_max=600)
        probe1_lowpass = si.resample(probe1_lowpass, 1250)

        #phase shift correction - equivalent to T-SHIFT in catGT
        probe1_phase_shift = si.phase_shift(probe1_lowpass)
        probe1_common_reference = si.common_reference(probe0_phase_shift,operator='median',reference='global')
        probe1_preprocessed = probe1_common_reference
        print('probe1_preprocessed',probe1_preprocessed)
    
        '''save preprocessed bin file before sorting'''
        #after saving, sorters will read this preprocessed binary file instead
        save_folder = dst_folder
        # print(save_folder)

        # file_path = save_folder+"imec0_preprocessed_LFP.bin"
        # with open(file_path, "wb") as file:
        #     #  Example binary data
        #     file.write(probe0_preprocessed)
        #     file.close()

        # file_path = save_folder+"imec1_preprocessed_LFP.bin"
        # with open(file_path, "wb") as file:
        #     #  Example binary data
        #     file.write(probe1_preprocessed)
        #     file.close()
        #si.write_binary_recording(recording=probe0_preprocessed, file_path=save_folder+'probe0_preprocessed_lf.bin', **job_kwargs)
        #si.write_binary_recording(recording=probe1_preprocessed, file_path=save_folder+'probe1_preprocessed_lf.bin', **job_kwargs)
        #probe0_preprocessed.save(folder=save_folder+'probe0_preprocessed', format='binary', **job_kwargs)
        probe1_preprocessed.save(folder=save_folder+'probe1_preprocessed', format='binary', **job_kwargs)
        print(g_file,'Done')
        probe0_preprocessed.save(folder=save_folder+'probe0_preprocessed', format='binary', **job_kwargs)
        print(g_file,'Done')

print('All Done! Overall it took:')

print(datetime.now() - startTime)
print('to finish!')
