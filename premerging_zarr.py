
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
import docker
from datetime import datetime
import itertools

import scipy.io as sio
startTime = datetime.now()
print('Start Time:' + startTime.strftime("%m/%d/%Y, %H:%M:%S"))
''' this section defines the animal and dates and fetch the recordings from the server to Beast'''
import sys
# The first command-line argument after the script name is the mouse identifier.
mouse = sys.argv[1]
# All command-line arguments after `mouse` and before `save_date` are considered dates.
dates = sys.argv[2].split(',')   # This captures all dates as a list.
# The last command-line argument is `save_date`.
save_date = sys.argv[3]
local_folder = sys.argv[4]
no_probe = sys.argv[5]
print(mouse)
print('acquisition folder: ',dates)
print(save_date)
use_ks4 = sys.argv[6].lower() in ['true', '1', 't', 'y', 'yes']
use_ks3 = sys.argv[7].lower() in ['true', '1', 't', 'y', 'yes']
base_folder = '/mnt/rds01/ibn-vision/DATA/SUBJECTS/'
save_folder = local_folder + save_date +'/'
# get all the recordings on that day

import os
import subprocess
subprocess.run('ulimit -n 4096',shell=True)
def sorting_key(s):
    return int(s.split('_g')[-1])

#grab recordings from the server to local machine (Beast)


job_kwargs = dict(n_jobs=32, chunk_duration='1s', progress_bar=True)

print(dates)
g_files_all = []
# iterate over all directories in source folder
date_count = 0

for probe in range(int(no_probe)):
    date_count = 0
    probe0_start_sample_fames = []
    probe0_end_sample_frames = []
    for date in dates:
        date_count = date_count + 1
        probe_name = 'imec' + str(probe) + '.ap'
        dst_folder = local_folder + date + '/'
        probe0_raw = si.read_zarr(dst_folder + 'probe' + str(probe) + '_raw_compressed.zarr')
        print(probe0_raw)


            

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

        
        if date_count == 1:
            probe0_cat_all = probe0_cat

        else:
            probe0_cat_all = si.concatenate_recordings([probe0_cat_all,probe0_cat])

    bad_channel_ids, channel_labels = si.detect_bad_channels(probe0_cat_all)
    probe0_cat_all = probe0_cat_all.remove_channels(bad_channel_ids)
    print('probe0_bad_channel_ids',bad_channel_ids)
    '''Motion Drift Correction'''
    #motion correction if needed
    #this is nonrigid correction - need to do parallel computing to speed up
    #assign parallel processing as job_kwargs

    #probe0_nonrigid_accurate = si.correct_motion(recording=probe0_cat_all, preset="kilosort_like",**job_kwargs)

    print('Start to motion correction finished:')
    print(datetime.now() - startTime)
    #kilosort like to mimic kilosort - no need if you are just running kilosort
    # probe0_kilosort_like = correct_motion(recording=probe0_cat, preset="kilosort_like")
    # probe1_kilosort_like = correct_motion(recording=probe1_cat, preset="kilosort_like")

    '''save preprocessed bin file before sorting'''


    #after saving, sorters will read this preprocessed binary file instead
    probe0_preprocessed_corrected = probe0_cat_all.save(folder=save_folder+'probe'+str(probe)+'_preprocessed', format='binary', **job_kwargs)
    print('Start to prepocessed files saved:')
    print(datetime.now() - startTime)
    #probe0_preprocessed_corrected = si.load_extractor(save_folder+'/probe0_preprocessed')
    #probe0_preprocessed_corrected = si.load_extractor(save_folder+'/probe1_preprocessed')
    ''' prepare sorters - currently using the default parameters and motion correction is turned off as it was corrected already above
        you can check if the parameters using:
        params = get_default_sorter_params('kilosort3')
    print("Parameters:\n", params)

    desc = get_sorter_params_description('kilosort3')
    print("Descriptions:\n", desc)

    Beware that moutainsort5 is commented out as the sorter somehow stops midway with no clue - currently raising this issue on their github page
    '''
    import pandas as pd
    def save_spikes_to_csv(spikes,save_folder):
        unit_index = spikes['unit_index']
        segment_index = spikes['segment_index']
        sample_index = spikes['sample_index']
        spikes_df = pd.DataFrame({'unit_index':unit_index,'segment_index':segment_index,'sample_index':sample_index})
        spikes_df.to_csv(save_folder + 'spikes.csv',index=False)

    #probe0_sorting_ks2_5 = si.run_sorter(sorter_name= 'kilosort2_5',recording=probe0_preprocessed_corrected,output_folder=dst_folder+'probe0/sorters/kilosort2_5/',docker_image="spikeinterface/kilosort2_5-compiled-base:latest",do_correction=False)
    #probe1_sorting_ks2_5 = si.run_sorter(sorter_name= 'kilosort2_5',recording=probe1_preprocessed_corrected,output_folder=dst_folder+'probe1/sorters/kilosort2_5/',docker_image="spikeinterface/kilosort2_5-compiled-base:latest",do_correction=False)
    #probe0_sorting_ks3 = si.run_sorter(sorter_name= 'kilosort3',recording=probe0_preprocessed_corrected,output_folder=dst_folder+'probe0/sorters/kilosort3/',docker_image="spikeinterface/kilosort3-compiled-base:latest",do_correction=False)
    #probe1_sorting_ks3 = si.run_sorter(sorter_name= 'kilosort3',recording=probe1_preprocessed_corrected,output_folder=dst_folder+'probe1/sorters/kilosort3/',docker_image="spikeinterface/kilosort3-compiled-base:latest",do_correction=False)
      # probe0_sorting_ks3 = si.remove_duplicated_spikes(sorting = probe0_sorting_ks3, censored_period_ms=0.3,method='keep_first')
    # probe1_sorting_ks2_5 = si.remove_duplicated_spikes(sorting = probe1_sorting_ks2_5, censored_period_ms=0.3,method='keep_first')
    # probe1_sorting_ks3 = si.remove_duplicated_spikes(sorting = probe1_sorting_ks3, censored_period_ms=0.3,method='keep_first')
    if use_ks4:
        probe0_sorting_ks4 = si.run_sorter(sorter_name= 'kilosort4',recording=probe0_preprocessed_corrected,output_folder=save_folder+'probe'+str(probe)+'/sorters/kilosort4/',docker_image='spikeinterface/kilosort4-base:latest')
        probe0_sorting_ks4 = si.remove_duplicated_spikes(sorting = probe0_sorting_ks4, censored_period_ms=0.3,method='keep_first')
        probe0_we_ks4 = si.create_sorting_analyzer(probe0_sorting_ks4, probe0_preprocessed_corrected, 
                                format = 'binary_folder',folder=save_folder +'probe'+str(probe)+'/waveform/kilosort4',
                                sparse = True,overwrite = True,
                                **job_kwargs)
        probe0_we_ks4.compute('random_spikes')
        probe0_we_ks4.compute('waveforms',ms_before=1.0, ms_after=2.0,**job_kwargs)
        probe0_ks4_spikes = np.load(save_folder + 'probe'+str(probe)+'/sorters/kilosort4/in_container_sorting/spikes.npy')
        save_spikes_to_csv(probe0_ks4_spikes,save_folder + 'probe'+str(probe)+'/sorters/kilosort4/in_container_sorting/')
    if use_ks3:
        probe0_sorting_ks3 = si.run_sorter(sorter_name= 'kilosort3',recording=probe0_preprocessed_corrected,output_folder=save_folder+'probe'+str(probe)+'/sorters/kilosort3/',docker_image='spikeinterface/kilosort3-compiled-base:latest')
        probe0_sorting_ks3 = si.remove_duplicated_spikes(sorting = probe0_sorting_ks3, censored_period_ms=0.3,method='keep_first')
        probe0_we_ks3 = si.create_sorting_analyzer(probe0_sorting_ks3, probe0_preprocessed_corrected, 
                                format = 'binary_folder',folder=save_folder +'probe'+str(probe)+'/waveform/kilosort3',
                                sparse = True,overwrite = True,
                                **job_kwargs)
        probe0_we_ks3.compute('random_spikes')
        probe0_we_ks3.compute('waveforms',ms_before=1.0, ms_after=2.0,**job_kwargs)
        probe0_ks3_spikes = np.load(save_folder + 'probe'+str(probe)+'/sorters/kilosort3/in_container_sorting/spikes.npy')
        save_spikes_to_csv(probe0_ks3_spikes,save_folder + 'probe'+str(probe)+'/sorters/kilosort3/in_container_sorting/')
    print('Start to all sorting done:')
    print(datetime.now() - startTime)




    ''' read sorters directly from the output folder - so you dont need to worry if something went wrong and you can't access the temp variables
        This section reads sorter outputs and extract waveforms 
    '''

sys.exit(0)