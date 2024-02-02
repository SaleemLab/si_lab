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
import pandas as pd
import itertools
base_folder = 'Z:\\ibn-vision\\DATA\\SUBJECTS\\'
mouse = 'M23038'
dates = ['20230816']
for date in dates:
    ephys_folder = base_folder + mouse + '\\ephys\\' + date +'\\'
    g_files = []
    # iterate over all directories in source folder

    for dirname in os.listdir(ephys_folder):
        # check if '_g' is in the directory name
        #only grab recording folders - there might be some other existing folders for analysis or sorted data
        if '_g' in dirname:
            # construct full directory path
            g_files.append(dirname)

            

    # Define a custom sorting key that extracts the number after 'g'
    def sorting_key(s):
        return int(s.split('_g')[-1])

    # Sort the list using the custom sorting key
    g_files = sorted(g_files, key=sorting_key)

    print(g_files) 
    #load first probe from beast folder - MEC probe for Diao
    probe0_raw = si.read_spikeglx(ephys_folder,stream_name='imec0.ap')
    print(probe0_raw)
    #Load second probe - V1 probe
    probe1_raw = si.read_spikeglx(ephys_folder,stream_name = 'imec1.ap')
    print(probe1_raw)
    probe0_num_segments = [probe0_raw.get_num_frames(segment_index=i) for i in range(probe0_raw.get_num_segments())]
    probe1_num_segments = [probe1_raw.get_num_frames(segment_index=i) for i in range(probe0_raw.get_num_segments())]
    probe0_end_sample_frames = list(itertools.accumulate(probe0_num_segments))
    probe0_start_sample_frames = [1] + [probe0_end_sample_frames[i] + 1 for i in range(0, len(probe0_num_segments)-1)]
    probe1_end_sample_frames = list(itertools.accumulate(probe1_num_segments))
    probe1_start_sample_frames = [1] + [probe1_end_sample_frames[i] + 1 for i in range(0, len(probe1_num_segments)-1)]

    print(probe0_start_sample_frames)
    print(probe1_start_sample_frames)
    print(probe0_end_sample_frames)
    print(probe1_end_sample_frames)

    probe0_segment_frames = pd.DataFrame({'segment_info':g_files,'segment start frame': probe0_start_sample_frames, 'segment end frame': probe0_end_sample_frames})
    probe0_segment_frames.to_csv(ephys_folder+'probe0\\sorters\\segment_frames.csv', index=False)
    probe1_segment_frames = pd.DataFrame({'segment_info':g_files,'segment start frame': probe1_start_sample_frames, 'segment end frame': probe1_end_sample_frames})
    probe1_segment_frames.to_csv(ephys_folder+'probe1\\sorters\\segment_frames.csv', index=False)