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
from spikeinterface.exporters import export_report
import spikeinterface.curation
import spikeinterface.widgets 
import docker
from datetime import datetime
#load mat file with merge suggestions
job_kwargs = dict(n_jobs=32, chunk_duration='1s', progress_bar=True)
base_folder = '/mnt/rds01/ibn-vision/DATA/SUBJECTS/'
mouse = 'M23034'
dates = ['20230807']
for date in dates:
    ephys_folder = base_folder + mouse + '/ephys/' + date +'/'
    analysis_folder = base_folder + mouse + '/analysis/' + date +'/'
    save_folder = base_folder + mouse + '/ephys/' + date +'/'
            
    
    probe0_sorting_ks3 = si.read_sorter_folder(ephys_folder + 'probe0/sorters/kilosort3_merged/')
    probe0_we_ks3 = si.load_sorting_analyzer(folder = ephys_folder + 'probe0/waveform/kilosort3_merged/') 
    probe0_preprocessed_corrected = si.load_extractor(ephys_folder + 'probe0_preprocessed/')
    export_report(sorting_analyzer = probe0_we_ks3, output_folder = ephys_folder + 'probe0/waveform/kilosort3_merged_report/')
    probe1_sorting_ks3 = si.read_sorter_folder(ephys_folder + 'probe1/sorters/kilosort3/')
    probe1_we_ks3 = si.load_sorting_analyzer(folder = ephys_folder + 'probe1/waveform/kilosort3/') 
    probe1_preprocessed_corrected = si.load_extractor(ephys_folder + 'probe1_preprocessed/')
    export_report(sorting_analyzer = probe1_we_ks3, output_folder = ephys_folder + 'probe1/waveform/kilosort3_merged_report/')
    
    

