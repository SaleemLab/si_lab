
# get all the recordings on that day
# allocate destination folder and move the ephys folder on the server to Beast lab user
from pathlib import Path

import os
import shutil

import numpy as np
import glob
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
import ast
import scipy.io as sio
import pandas as pd


startTime = datetime.now()
print('Start Time:' + startTime.strftime("%m/%d/%Y, %H:%M:%S"))
''' this section defines the animal and dates and fetch the recordings from the server to Beast'''
import sys
# The first command-line argument after the script name is the mouse identifier.
mouse = sys.argv[1]
# All command-line arguments after `mouse` and before `save_date` are considered dates.

# The last command-line argument is `save_date`.
save_date = sys.argv[2]
local_folder = sys.argv[3]
no_probe = sys.argv[4]
print(mouse)
print(save_date)
use_ks4 = sys.argv[5].lower() in ['true', '1', 't', 'y', 'yes']
use_ks3 = sys.argv[6].lower() in ['true', '1', 't', 'y', 'yes']
base_folder = '/saleem/ibn-vision/DATA/SUBJECTS/'
save_folder = local_folder + save_date +'/'

# Check g files to ignore are correct (tcat should always be ignored)
# Check if sys.argv[8] is empty
if len(sys.argv) > 7 and sys.argv[7]:
    g_files_to_ignore = ast.literal_eval(sys.argv[7])
else:
    g_files_to_ignore = []

# Print the result to verify
print(f"g_files_to_ignore: {g_files_to_ignore}")
# get all the recordings on that day

import os
import subprocess
subprocess.run('ulimit -n 4096',shell=True)
def sorting_key(s):
    return int(s.split('_g')[-1])

#grab recordings from the server to local machine (Beast)


job_kwargs = dict(n_jobs=32, chunk_duration='1s', progress_bar=True)

g_files_all = []
# iterate over all directories in source folder
acquisition_base_path = base_folder + mouse + '/ephys/' + save_date + '/*' + save_date
acquisition_folders = glob.glob(acquisition_base_path + '_*')
acquisition_list = sorted([int(folder.split('_')[-1]) for folder in acquisition_folders])
date_count = 0

    
probes=[1]
for probe in probes:
    date_count = 0
    start_sample_frames = []
    end_sample_frames = []
    segment_info_all = []
    bad_channel_ids_all = np.array([])
    for acquisition in acquisition_list:
        
        probe_name = 'imec' + str(probe) + '.ap'
        dst_folder = local_folder + save_date + '/probe' + str(probe) + '_compressed_' + str(acquisition) + '.zarr'
        raw = si.read_zarr(dst_folder)
        print(raw)
        no_segments = raw.get_num_segments()
        if g_files_to_ignore:
            g_files_to_ignore_this_acquisition = g_files_to_ignore[date_count]
        else:
            g_files_to_ignore_this_acquisition = []
        segments = [i for i in range(no_segments) if i not in g_files_to_ignore_this_acquisition]
        num_segments = [raw.get_num_frames(segment_index=i) for i in segments]
        segment_info = [str(acquisition) + '_g' + str(i) for i in segments]
        segment_info_all = segment_info_all + segment_info
        end_sample_frames_tmp = list(itertools.accumulate(num_segments))
        if date_count == 0:
            start_sample_frames = [1] + [end_sample_frames_tmp[i] + 1 for i in range(0, len(num_segments)-1)]
            end_sample_frames = end_sample_frames + end_sample_frames_tmp
        else:
            start_sample_frames = start_sample_frames + [end_sample_frames[-1]+1] + \
            [end_sample_frames[-1]+end_sample_frames_tmp[i] + 1 for i in range(0, len(num_segments)-1)]
            end_sample_frames = end_sample_frames + [end_sample_frames_tmp[i] + end_sample_frames[-1] for i in range(0, len(num_segments))]
            
        

            
        #select segments if needed
        
        raw_selected = si.select_segment_recording(raw,segment_indices=segments)
        
        
        #several preprocessing steps and concatenation of the recordings
        #highpass filter - threhsold at 300Hz
        highpass = si.highpass_filter(raw_selected,freq_min=300.)
        #phase shift correction - equivalent to T-SHIFT in catGT
        phase_shift = si.phase_shift(highpass)
        common_reference = si.common_reference(phase_shift,operator='median',reference='global')
        preprocessed = common_reference
        cat = si.concatenate_recordings([preprocessed])
        print('preprocessed',preprocessed)
        print('concatenated',cat)
        bad_channel_ids, channel_labels = si.detect_bad_channels(cat)
        print('bad_channel_ids',bad_channel_ids,'in acquisition:',str(acquisition))
        bad_channel_ids_all = np.concatenate((bad_channel_ids_all,bad_channel_ids))
        print(cat)
        if date_count == 0:
            cat_all = cat

        else:
            cat_all = si.concatenate_recordings([cat_all,cat],sampling_frequency_max_diff=1.0)
            
        date_count = date_count + 1
    

    segment_frames = pd.DataFrame({'segment_info':segment_info_all,'segment start frame': start_sample_frames, 'segment end frame': end_sample_frames})
    segment_frames.to_csv(save_folder+'probe'+str(probe)+'segment_frames.csv', index=False)

    bad_channel_ids_all = np.unique(bad_channel_ids_all)
    print('total bad channel ids all',bad_channel_ids_all,'in probe:',str(probe),'in all acquisitions:',str(acquisition_list))
    cat_all = cat_all.remove_channels(bad_channel_ids_all)

    print('concatenated all recordings:',cat_all)
    print('Start to prepocessed files saved:')
    print(datetime.now() - startTime)
    #probe0_preprocessed_corrected = si.load_extractor(save_folder+'/probe0_preprocessed')
    preprocessed_corrected = si.load_extractor(save_folder+'probe1_preprocessed')
    ''' prepare sorters - currently using the default parameters and motion correction is turned off as it was corrected already above
        you can check if the parameters using:
        params = get_default_sorter_params('kilosort3')
    print("Parameters:\n", params)

    desc = get_sorter_params_description('kilosort3')
    print("Descriptions:\n", desc)'''

    import pandas as pd
    def save_spikes_to_csv(spikes,save_folder):
        unit_index = spikes['unit_index']
        segment_index = spikes['segment_index']
        sample_index = spikes['sample_index']
        spikes_df = pd.DataFrame({'unit_index':unit_index,'segment_index':segment_index,'sample_index':sample_index})
        spikes_df.to_csv(save_folder + 'spikes.csv',index=False)

    if use_ks4:
        sorting_ks4 = si.run_sorter(sorter_name= 'kilosort4',recording=preprocessed_corrected,output_folder=save_folder+'probe'+str(probe)+'/sorters/kilosort4/',docker_image='spikeinterface/kilosort4-base:latest',do_correction=False,use_binary_file=True,clear_cache=True)
        sorting_ks4 = si.remove_duplicated_spikes(sorting = sorting_ks4, censored_period_ms=0.3,method='keep_first')
        we_ks4 = si.create_sorting_analyzer(sorting_ks4, preprocessed_corrected, 
                                format = 'binary_folder',folder=save_folder +'probe'+str(probe)+'/waveform/kilosort4',
                                sparse = True,overwrite = True,
                                **job_kwargs)
        we_ks4.compute('random_spikes')
        we_ks4.compute('waveforms',ms_before=1.0, ms_after=2.0,**job_kwargs)
        ks4_spikes = np.load(save_folder + 'probe'+str(probe)+'/sorters/kilosort4/in_container_sorting/spikes.npy')
        save_spikes_to_csv(ks4_spikes,save_folder + 'probe'+str(probe)+'/sorters/kilosort4/in_container_sorting/')
        
    if use_ks3:
        sorting_ks3 = si.run_sorter(sorter_name= 'kilosort3',recording=preprocessed_corrected,output_folder=save_folder+'probe'+str(probe)+'/sorters/kilosort3/',docker_image='spikeinterface/kilosort3-compiled-base:latest',do_correction=False)
        sorting_ks3 = si.remove_duplicated_spikes(sorting = sorting_ks3, censored_period_ms=0.3,method='keep_first')
        we_ks3 = si.create_sorting_analyzer(sorting_ks3, preprocessed_corrected, 
                                format = 'binary_folder',folder=save_folder +'probe'+str(probe)+'/waveform/kilosort3',
                                sparse = True,overwrite = True,
                                **job_kwargs)
        we_ks3.compute('random_spikes')
        we_ks3.compute('waveforms',ms_before=1.0, ms_after=2.0,**job_kwargs)
        ks3_spikes = np.load(save_folder + 'probe'+str(probe)+'/sorters/kilosort3/in_container_sorting/spikes.npy')
        save_spikes_to_csv(ks3_spikes,save_folder + 'probe'+str(probe)+'/sorters/kilosort3/in_container_sorting/')
    print('Start to all sorting done:')
    print(datetime.now() - startTime)





    ''' read sorters directly from the output folder - so you dont need to worry if something went wrong and you can't access the temp variables
        This section reads sorter outputs and extract waveforms 
    '''

sys.exit(0)