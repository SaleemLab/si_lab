from datetime import datetime

startTime = datetime.now()
print('Start Time:' + startTime.strftime("%m/%d/%Y, %H:%M:%S"))
''' this section defines the animal and dates and fetch the recordings from the server to Beast'''

from pathlib import Path

import os
import shutil

import numpy as np

import spikeinterface.sorters
import spikeinterface.full as si
import scipy.signal
import spikeinterface.extractors as se
import spikeinterface.comparison
import spikeinterface.exporters
import spikeinterface.curation
import spikeinterface.widgets
import docker
import itertools

import scipy.io as sio

import sys

import os
import subprocess

subprocess.run('ulimit -n 4096', shell=True)


def sorting_key(s):
    return int(s.split('_g')[-1])


import pandas as pd


def save_spikes_to_csv(spikes, save_folder):
    unit_index = spikes['unit_index']
    segment_index = spikes['segment_index']
    sample_index = spikes['sample_index']
    spikes_df = pd.DataFrame({'unit_index': unit_index, 'segment_index': segment_index, 'sample_index': sample_index})
    spikes_df.to_csv(save_folder + 'spikes.csv', index=False)


# grab recordings from the server to local machine (Beast)

job_kwargs = dict(n_jobs=32, chunk_duration='1s', progress_bar=True)

# The first command-line argument after the script name is the mouse identifier.
# mouse='M24019' #mouse id
# save_date='20240716' #date of recording
# dates='20240716/20240716_0,20240716/20240716_2' #acquisition date and session e.g. dates='20240624/20240624_0,20240624/20240624_1'
# dates=dates.split(',')
# base_folder='/home/lab/spikeinterface_sorting/temp_data/'  # Adjust this path if necessary
# local_folder = base_folder
# no_probe=1 #number of probes you have in this session

# The first command-line argument after the script name is the mouse identifier.
mouse = sys.argv[1]
# All command-line arguments after `mouse` and before `save_date` are considered dates.
dates = sys.argv[2].split(',')  # This captures all dates as a list.
# The last command-line argument is `save_date`.
save_date = sys.argv[3]
local_folder = sys.argv[4]
no_probe = sys.argv[5]
print(mouse)
print('acquisition folder: ', dates)
use_ks4 = sys.argv[6].lower() in ['true', '1', 't', 'y', 'yes']
use_ks3 = sys.argv[7].lower() in ['true', '1', 't', 'y', 'yes']
base_folder = '/mnt/rds01/ibn-vision/DATA/SUBJECTS/'

save_folder = local_folder + mouse + "/"
print('save folder: ', save_folder)

# get the output folder of CatGT for SI to read
nAcq = (len(dates))

if nAcq == 1:
    date = dates[0]
    runName = date.split('/')
    baseDate = runName[0]
    tempDates = dates[0].split('/')
    outDir = save_folder + 'ephys' + '/' + dates[0] + '/' + 'catgt_' + runName[1] + '_g0/'
    print('Final concatenated file: ')
    print(outDir)
    save_folder = outDir

if nAcq > 1:
    date = dates[0]
    runName = date.split('/')
    baseDate = runName[0]
    tempDates = dates[0].split('/')
    outDir = save_folder + baseDate + '/' + 'supercat_' + mouse + '_' + tempDates[1] + '_g0/'
    print('Final concatenated file: ')
    print(outDir)
    save_folder = outDir

for probe in range(int(no_probe)):

    # load the probe
    print('probe #: ', probe)
    probe_name = 'imec' + str(probe) + '.ap'
    probe_raw = si.read_spikeglx(outDir, stream_name=probe_name)
    print(probe_raw)

    # pre-processing steps
    # highpass filter - threhsold at 300Hz
    probe_highpass = si.highpass_filter(probe_raw, freq_min=300.)

    # detect and remove bad channels
    bad_channel_ids, channel_labels = si.detect_bad_channels(probe_highpass)
    probe_bad_channels = probe_highpass.remove_channels(bad_channel_ids)
    print('probe_bad_channel_ids', bad_channel_ids)

    # phase shift correction - equivalent to T-SHIFT in catGT
    probe_phase_shift = si.phase_shift(probe_bad_channels)
    probe_common_reference = si.common_reference(probe_phase_shift, operator='median', reference='global')

    probe0_preprocessed_corrected = probe_common_reference
    probe0_preprocessed_corrected = probe_raw
    # print(probe0_preprocessed_corrected)

    # save pre-processed catenated file
    # probe0_preprocessed_corrected = probe_common_reference.save(folder=save_folder+'probe'+str(probe)+'_preprocessed', format='binary', **job_kwargs)

    # do the spike sorting

    if use_ks4:
        print('Running kilosort 4 on probe ', probe)
        probe0_sorting_ks4 = si.run_sorter(sorter_name='kilosort4', recording=probe0_preprocessed_corrected,
                                           folder=save_folder + 'probe' + str(probe) + '/sorters/kilosort4/',
                                           docker_image='spikeinterface/kilosort4-base:latest')
        probe0_sorting_ks4 = si.remove_duplicated_spikes(sorting=probe0_sorting_ks4, censored_period_ms=0.3,
                                                         method='keep_first')
        probe0_we_ks4 = si.create_sorting_analyzer(probe0_sorting_ks4, probe0_preprocessed_corrected,
                                                   format='binary_folder',
                                                   folder=save_folder + 'probe' + str(probe) + '/waveform/kilosort4',
                                                   sparse=True, overwrite=True,
                                                   **job_kwargs)
        probe0_we_ks4.compute('random_spikes')
        probe0_we_ks4.compute('waveforms', ms_before=1.0, ms_after=2.0, **job_kwargs)
        probe0_ks4_spikes = np.load(
            save_folder + 'probe' + str(probe) + '/sorters/kilosort4/in_container_sorting/spikes.npy')
        save_spikes_to_csv(probe0_ks4_spikes,
                           save_folder + 'probe' + str(probe) + '/sorters/kilosort4/in_container_sorting/')
    if use_ks3:
        print('Running kilosort 3 on probe ', probe)
        # probe0_sorting_ks3 = si.run_sorter(sorter_name= 'kilosort3',recording=probe0_preprocessed_corrected,folder=save_folder+'probe'+str(probe)+'/sorters/kilosort3/',docker_image='spikeinterface/kilosort3-compiled-base:latest')
        probe0_sorting_ks3 = si.run_sorter(sorter_name='kilosort3', recording=probe0_preprocessed_corrected,
                                           folder=save_folder + 'probe' + str(probe) + '/sorters/kilosort3/',
                                           docker_image=True, verbose=True)

        probe0_sorting_ks3 = si.remove_duplicated_spikes(sorting=probe0_sorting_ks3, censored_period_ms=0.3,
                                                         method='keep_first')
        probe0_we_ks3 = si.create_sorting_analyzer(probe0_sorting_ks3, probe0_preprocessed_corrected,
                                                   format='binary_folder',
                                                   folder=save_folder + 'probe' + str(probe) + '/waveform/kilosort3',
                                                   sparse=True, overwrite=True,
                                                   **job_kwargs)
        probe0_we_ks3.compute('random_spikes')
        probe0_we_ks3.compute('waveforms', ms_before=1.0, ms_after=2.0, **job_kwargs)
        probe0_ks3_spikes = np.load(
            save_folder + 'probe' + str(probe) + '/sorters/kilosort3/in_container_sorting/spikes.npy')
        save_spikes_to_csv(probe0_ks3_spikes,
                           save_folder + 'probe' + str(probe) + '/sorters/kilosort3/in_container_sorting/')

sys.exit(0)