
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

# Check g files to ignore are correct (tcat should always be ignored)
g_files_to_ignore = ['tcat','0_g6','0_g7','0_g8','0_g9','1_g','lf.bin','if.meta']
#g_files_to_ignore = sys.argv[8]
print(g_files_to_ignore)
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
for date in dates:
    print('acquisition folder:',date)
    date_count = date_count + 1
    ephys_folder = base_folder + mouse + '/ephys/' + date +'/'
    dst_folder = local_folder + date + '/'
    ephys_folder = base_folder + mouse + '/ephys/' + date +'/'
    g_files = []

    # Rename all tcat files to t0 if they exist
    tcat_pattern = os.path.join(ephys_folder,'**','*tcat.imec*.lf*')
    files_to_rename = glob.glob(tcat_pattern, recursive=True)
    # Step 1: Iterate over the list of files with tcat in the name
    for old_name in files_to_rename:
        # Step 2: Construct the new filename (REMEMBER to switch the name back to tcat)
        new_name = old_name.replace('tcat', 't0')
        
        # Step 3: Rename the file
        os.rename(old_name, new_name)
        print(f'Renamed {old_name} to {new_name}')

    print('copying ephys data from:' + ephys_folder)
    for dirname in os.listdir(ephys_folder):
        # ignore some folders or files includiing tcat and other G files (such as checkerboard recordings)
        if any(ignore_str in dirname for ignore_str in g_files_to_ignore):
            continue
    #     # check if '_g' is in the directory name
    #     #only grab recording folders - there might be some other existing folders for analysis or sorted data
        if '_g' in dirname:
    #         # construct full directory path
            g_files.append(dirname)
            source = os.path.join(ephys_folder, dirname)
            destination = os.path.join(dst_folder, dirname)
            # copy the directory to the destination folder
            shutil.copytree(source, destination)
    print('Start to copying files to local folder: ')
    print(datetime.now() - startTime)
    ''' read spikeglx recordings and preprocess them'''
    # Define a custom sorting key that extracts the number after 'g'

    # Sort the list using the custom sorting key
    g_files = sorted(g_files, key=sorting_key)
    g_files_all = g_files_all + g_files
    print(g_files)
    print('all g files:',g_files_all) 
    # stream_names, stream_ids = si.get_neo_streams('spikeglx',dst_folder)
    # print(stream_names)
    # print(stream_ids)
    
for probe in range(int(no_probe)):
    date_count = 0
    probe0_start_sample_fames = []
    probe0_end_sample_frames = []
    for date in dates:
        date_count = date_count + 1
        probe_name = 'imec' + str(probe) + '.ap'
        dst_folder = local_folder + date + '/'
        probe0_raw = si.read_spikeglx(dst_folder,stream_name=probe_name)
        print(probe0_raw)


        probe0_num_segments = [probe0_raw.get_num_frames(segment_index=i) for i in range(probe0_raw.get_num_segments())]
        
        probe0_end_sample_frames_tmp = list(itertools.accumulate(probe0_num_segments))
        if date_count == 1:
            probe0_start_sample_frames = [1] + [probe0_end_sample_frames_tmp[i] + 1 for i in range(0, len(probe0_num_segments)-1)]
            probe0_end_sample_frames = probe0_end_sample_frames + probe0_end_sample_frames_tmp
        else:
            probe0_start_sample_frames = probe0_start_sample_frames + [probe0_end_sample_frames[-1]+1] + \
            [probe0_end_sample_frames[-1]+probe0_end_sample_frames_tmp[i] + 1 for i in range(0, len(probe0_num_segments)-1)]
            probe0_end_sample_frames = probe0_end_sample_frames + [probe0_end_sample_frames_tmp[i] + probe0_end_sample_frames[-1] for i in range(0, len(probe0_num_segments))]

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
    probe0_cat_all = probe0_cat_all.astype(np.float32)
    si.correct_motion(recording=probe0_cat_all, preset="nonrigid_accurate",folder=save_folder+'probe'+str(probe)+'_motion',output_motion_info=True,**job_kwargs)
    #probe0_cat_all = si.correct_motion(recording=probe0_cat_all, preset="nonrigid_accurate",folder=save_folder+'probe'+str(probe)+'_motion',output_motion_info=True,**job_kwargs)
    
    # not sure why the error happens if save probe0_cat_all directly after motion correction
    # But it works if load motion info then interpolate motion then save, it works 
    motion_info = si.load_motion_info(save_folder+'probe'+str(probe)+'_motion')

    from spikeinterface.sortingcomponents.motion import estimate_motion, interpolate_motion
    probe0_motion_corrected = interpolate_motion(
                    recording=probe0_cat_all,
                    motion=motion_info['motion'],
                    **motion_info['parameters']['interpolate_motion_kwargs'])

    probe0_cat_all = probe0_motion_corrected
    # Back to int16 to save space
    probe0_cat_all = probe0_cat_all.astype(np.int16)

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
    #probe0_preprocessed_corrected = si.load_extractor(save_folder+'probe'+str(probe)+'_preprocessed')
    #probe0_preprocessed_corrected = si.load_extractor(save_folder+'/probe1_preprocessed')

    print('Start to prepocessed files saved:')
    print(datetime.now() - startTime)
    #probe0_preprocessed_corrected = si.load_extractor(save_folder+'probe'+str(probe)+'_preprocessed')
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
        # Use docker KS4
        probe0_preprocessed_corrected = si.load_extractor(save_folder+'probe'+str(probe)+'_preprocessed')
        probe0_sorting_ks4 = si.run_sorter(sorter_name= 'kilosort4',recording=probe0_preprocessed_corrected,output_folder=save_folder+'probe'+str(probe)+'/sorters/kilosort4/',docker_image='spikeinterface/kilosort4-base:latest',do_correction=False)
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
        # Use local KS3
        si.Kilosort3Sorter.set_kilosort3_path('/home/saleem_lab/Kilosort')
        probe0_preprocessed_corrected = si.load_extractor(save_folder+'probe'+str(probe)+'_preprocessed')
        probe0_sorting_ks3 = si.run_sorter(sorter_name= 'kilosort3',recording=probe0_preprocessed_corrected,output_folder=save_folder+'probe'+str(probe)+'/sorters/kilosort3/',do_correction=False)
        probe0_sorting_ks3 = si.remove_duplicated_spikes(sorting = probe0_sorting_ks3, censored_period_ms=0.3,method='keep_first')
        probe0_we_ks3 = si.create_sorting_analyzer(probe0_sorting_ks3, probe0_preprocessed_corrected, 
                                format = 'binary_folder',folder=save_folder +'probe'+str(probe)+'/waveform/kilosort3',
                                sparse = True,overwrite = True,
                                **job_kwargs)
        probe0_we_ks3.compute('random_spikes')
        probe0_we_ks3.compute('waveforms',ms_before=1.0, ms_after=2.0,**job_kwargs)
        probe0_ks3_spikes = np.load(save_folder + 'probe'+str(probe)+'/waveform/kilosort3/sorting/spikes.npy') # testing waveform folder rather than sorters folder for reading spikes.npy file
        #probe0_ks3_spikes = np.load(save_folder + 'probe'+str(probe)+'/sorters/kilosort3/in_container_sorting/spikes.npy')

        if not os.path.exists(save_folder + 'probe'+str(probe)+'/sorters/kilosort3/in_container_sorting/'): #if missing in_container_sorting folder, create one just for saving spike.csv in it
            os.makedirs(save_folder + 'probe'+str(probe)+'/sorters/kilosort3/in_container_sorting/')
            shutil.copyfile(save_folder + 'probe'+str(probe)+'/waveform/kilosort3/sorting/spikes.npy', save_folder + 'probe'+str(probe)+'/sorters/kilosort3/in_container_sorting/spikes.npy')
        save_spikes_to_csv(probe0_ks3_spikes,save_folder + 'probe'+str(probe)+'/waveform/kilosort3/sorting/')
        save_spikes_to_csv(probe0_ks3_spikes,save_folder + 'probe'+str(probe)+'/sorters/kilosort3/in_container_sorting/')
    print('Start to all sorting done:')
    print(datetime.now() - startTime)

    import pandas as pd
    probe0_segment_frames = pd.DataFrame({'segment_info':g_files_all,'segment start frame': probe0_start_sample_frames, 'segment end frame': probe0_end_sample_frames})
    probe0_segment_frames.to_csv(save_folder+'probe'+str(probe)+'/sorters/segment_frames.csv', index=False)

    #process to change all the folder paths in text and .json files on Beast to the server before uploading it to the server
    import os
    import glob

    # Define the folder list
    folder_list = [save_folder + 'probe'+str(probe)+'_preprocessed', 
                save_folder + 'probe'+str(probe)+'/waveform/',
                save_folder + 'probe'+str(probe)+'/sorters/']

    temp_wh_files = []
    # Go through each folder in the folder list
    for folder in folder_list:
        # Recursively find all temp files in the folder and its subfolders
        for temp_wh_file in glob.glob(os.path.join(folder, '**', 'temp_wh.dat'), recursive=True):
            # Append the found temp file path to the list
            temp_wh_files.append(temp_wh_file)

    for files in temp_wh_files:
        os.remove(files)

    ''' read sorters directly from the output folder - so you dont need to worry if something went wrong and you can't access the temp variables
        This section reads sorter outputs and extract waveforms 
    '''

sys.exit(0)