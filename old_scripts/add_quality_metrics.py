
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
mouse = 'M23034'
dates = ['20230804','20230805','20230806','20230807']

job_kwargs = dict(n_jobs=32, chunk_duration='1s', progress_bar=True)
# ephys folder
for date in dates:
    ephys_folder = base_folder + mouse + '/ephys/' + date +'/'
    print(ephys_folder)

    probe1_we_ks4 = si.load_waveforms(folder = ephys_folder + 'probe1/waveform/kilosort4/')

    qm_list = si.get_quality_metric_list()
    print('The following quality metrics are computed:')
    print(qm_list)

    probe1_ks4_metrics = si.compute_quality_metrics(probe1_we_ks4, metric_names=qm_list,**job_kwargs)
    
    
    
