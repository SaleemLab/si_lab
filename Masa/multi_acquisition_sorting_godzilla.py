
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
mouse = 'M24017'
dates = ['20240608/20240608_0','20240608/20240608_1']
save_date = '20240608'
base_folder = '/mnt/rds01/ibn-vision/DATA/SUBJECTS/'
save_base_folder = "/home/lab/spikeinterface_sorting/temp_data/"
save_folder = save_base_folder +save_date+'/'
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


job_kwargs = dict(n_jobs=32, chunk_duration='1s', progress_bar=True)


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


#after saving, sorters will read this preprocessed binary file instead
probe0_preprocessed_corrected = probe0_nonrigid_accurate.save(folder=save_folder+'probe0_preprocessed', format='binary', **job_kwargs)
probe1_preprocessed_corrected = probe1_nonrigid_accurate.save(folder=save_folder+'probe1_preprocessed', format='binary', **job_kwargs)
print('Start to prepocessed files saved:')
print(datetime.now() - startTime)
#probe0_preprocessed_corrected = si.load_extractor(save_folder+'/probe0_preprocessed')
#probe1_preprocessed_corrected = si.load_extractor(save_folder+'/probe1_preprocessed')
''' prepare sorters - currently using the default parameters and motion correction is turned off as it was corrected already above
    you can check if the parameters using:
    params = get_default_sorter_params('kilosort3')
print("Parameters:\n", params)

desc = get_sorter_params_description('kilosort3')
print("Descriptions:\n", desc)

Beware that moutainsort5 is commented out as the sorter somehow stops midway with no clue - currently raising this issue on their github page
'''
#probe0_sorting_ks2_5 = si.run_sorter(sorter_name= 'kilosort2_5',recording=probe0_preprocessed_corrected,output_folder=dst_folder+'probe0/sorters/kilosort2_5/',docker_image="spikeinterface/kilosort2_5-compiled-base:latest",do_correction=False)
#probe1_sorting_ks2_5 = si.run_sorter(sorter_name= 'kilosort2_5',recording=probe1_preprocessed_corrected,output_folder=dst_folder+'probe1/sorters/kilosort2_5/',docker_image="spikeinterface/kilosort2_5-compiled-base:latest",do_correction=False)
#probe0_sorting_ks3 = si.run_sorter(sorter_name= 'kilosort3',recording=probe0_preprocessed_corrected,output_folder=dst_folder+'probe0/sorters/kilosort3/',docker_image="spikeinterface/kilosort3-compiled-base:latest",do_correction=False)
#probe1_sorting_ks3 = si.run_sorter(sorter_name= 'kilosort3',recording=probe1_preprocessed_corrected,output_folder=dst_folder+'probe1/sorters/kilosort3/',docker_image="spikeinterface/kilosort3-compiled-base:latest",do_correction=False)
probe0_sorting_ks4 = si.run_sorter(sorter_name= 'kilosort4',recording=probe0_preprocessed_corrected,folder=save_folder+'probe0/sorters/kilosort4/',docker_image='spikeinterface/kilosort4-base:latest',do_correction=False)
probe1_sorting_ks4 = si.run_sorter(sorter_name= 'kilosort4',recording=probe1_preprocessed_corrected,folder=save_folder+'probe1/sorters/kilosort4/',docker_image='spikeinterface/kilosort4-base:latest',do_correction=False)
probe0_sorting_ks3 = si.run_sorter(sorter_name= 'kilosort3',recording=probe0_preprocessed_corrected,folder=save_folder+'probe0/sorters/kilosort3/',docker_image='spikeinterface/kilosort3-compiled-base:latest',do_correction=False)
probe1_sorting_ks3 = si.run_sorter(sorter_name= 'kilosort3',recording=probe1_preprocessed_corrected,folder=save_folder+'probe1/sorters/kilosort3/',docker_image='spikeinterface/kilosort3-compiled-base:latest',do_correction=False)
#remove duplicates
# probe0_sorting_ks2_5 = si.remove_duplicated_spikes(sorting = probe0_sorting_ks2_5, censored_period_ms=0.3,method='keep_first')
# probe0_sorting_ks3 = si.remove_duplicated_spikes(sorting = probe0_sorting_ks3, censored_period_ms=0.3,method='keep_first')
# probe1_sorting_ks2_5 = si.remove_duplicated_spikes(sorting = probe1_sorting_ks2_5, censored_period_ms=0.3,method='keep_first')
# probe1_sorting_ks3 = si.remove_duplicated_spikes(sorting = probe1_sorting_ks3, censored_period_ms=0.3,method='keep_first')
probe0_sorting_ks4 = si.remove_duplicated_spikes(sorting = probe0_sorting_ks4, censored_period_ms=0.3,method='keep_first')
probe1_sorting_ks4 = si.remove_duplicated_spikes(sorting = probe1_sorting_ks4, censored_period_ms=0.3,method='keep_first')
probe0_sorting_ks3 = si.remove_duplicated_spikes(sorting = probe0_sorting_ks3, censored_period_ms=0.3,method='keep_first')
probe1_sorting_ks3 = si.remove_duplicated_spikes(sorting = probe1_sorting_ks3, censored_period_ms=0.3,method='keep_first')
print(probe0_sorting_ks4)
print(probe1_sorting_ks4)
print('Start to all sorting done:')
print(datetime.now() - startTime)

import pandas as pd
probe0_segment_frames = pd.DataFrame({'segment_info':g_files_all,'segment start frame': probe0_start_sample_frames, 'segment end frame': probe0_end_sample_frames})
probe0_segment_frames.to_csv(save_folder+'probe0/sorters/segment_frames.csv', index=False)
probe1_segment_frames = pd.DataFrame({'segment_info':g_files_all,'segment start frame': probe1_start_sample_frames, 'segment end frame': probe1_end_sample_frames})
probe1_segment_frames.to_csv(save_folder+'probe1/sorters/segment_frames.csv', index=False)


''' read sorters directly from the output folder - so you dont need to worry if something went wrong and you can't access the temp variables
    This section reads sorter outputs and extract waveforms 
'''

probe0_we_ks4 = si.ecreate_sorting_analyzer(probe0_preprocessed_corrected, probe0_sorting_ks4, folder=save_folder +'probe0/waveform/kilosort4',
                        sparse=True, max_spikes_per_unit=500, ms_before=1.5,ms_after=2.,
                        **job_kwargs)


probe1_we_ks4 = si.create_sorting_analyzer(probe1_preprocessed_corrected, probe1_sorting_ks4, folder=save_folder +'probe1/waveform/kilosort4',
                        sparse=True, max_spikes_per_unit=500, ms_before=1.5,ms_after=2.,
                        **job_kwargs)

probe0_we_ks3 = si.create_sorting_analyzer(probe0_preprocessed_corrected, probe0_sorting_ks3, folder=save_folder +'probe0/waveform/kilosort3',
                        sparse=True, max_spikes_per_unit=500, ms_before=1.5,ms_after=2.,
                        **job_kwargs)

probe1_we_ks3 = si.create_sorting_analyzer(probe1_preprocessed_corrected, probe1_sorting_ks3, folder=save_folder +'probe1/waveform/kilosort3',
                        sparse=True, max_spikes_per_unit=500, ms_before=1.5,ms_after=2.,
                        **job_kwargs)

probe0_we_ks4.compute('random_spikes',**job_kwargs)
probe1_we_ks4.compute('random_spikes',**job_kwargs)
probe0_we_ks3.compute('random_spikes',**job_kwargs)
probe1_we_ks3.compute('random_spikes',**job_kwargs)

probe0_we_ks4.compute('waveforms',ms_before=1.0, ms_after=2.0,**job_kwargs)
probe1_we_ks4.compute('waveforms',ms_before=1.0, ms_after=2.0,**job_kwargs)
probe0_we_ks3.compute('waveforms',ms_before=1.0, ms_after=2.0,**job_kwargs)
probe1_we_ks3.compute('waveforms',ms_before=1.0, ms_after=2.0,**job_kwargs)

probe0_ks4_spikes = np.load(save_folder + 'probe0/sorters/kilosort4/in_contianer_sorting/spikes.npy')
probe1_ks4_spikes = np.load(save_folder + 'probe1/sorters/kilosort4/in_contianer_sorting/spikes.npy')  
probe0_ks3_spikes = np.load(save_folder + 'probe0/sorters/kilosort3/in_contianer_sorting/spikes.npy')
probe1_ks3_spikes = np.load(save_folder + 'probe1/sorters/kilosort3/in_contianer_sorting/spikes.npy')
def save_spikes_to_csv(spikes,save_folder):
    unit_index = spikes['unit_index']
    segment_index = spikes['segment_index']
    sample_index = spikes['sample_index']
    spikes_df = pd.DataFrame({'unit_index':unit_index,'segment_index':segment_index,'sample_index':sample_index})
    spikes_df.to_csv(save_folder + 'spikes.csv',index=False)
    
save_spikes_to_csv(probe0_ks4_spikes,save_folder + 'probe0/sorters/kilosort4/in_contianer_sorting/')
save_spikes_to_csv(probe1_ks4_spikes,save_folder + 'probe1/sorters/kilosort4/in_contianer_sorting/')
save_spikes_to_csv(probe0_ks3_spikes,save_folder + 'probe0/sorters/kilosort3/in_contianer_sorting/')
save_spikes_to_csv(probe1_ks3_spikes,save_folder + 'probe1/sorters/kilosort3/in_contianer_sorting/')


analysis_folder = base_folder + mouse + '/analysis/' + save_date +'/'
merge_suggestions = sio.loadmat(analysis_folder + 'probe0um_merge_suggestion_ks4.mat')
match_ids = merge_suggestions['match_ids']
merge_ids = match_ids[:,1]
original_ids = match_ids[:,0]
cs_probe0 = si.CurationSorting(parent_sorting=probe0_sorting_ks4)
unique_ids = np.unique(merge_ids)
for id in unique_ids:
    id_count = np.count_nonzero(merge_ids == id)
    if id_count > 1:
        units = np.where(merge_ids == id)
        units_index = (units[0]-1,)
        cs_probe0.merge(original_ids[units_index])
        
probe0_sorting_ks4_merged = cs_probe0.sorting
probe0_sorting_ks4_merged.save(folder = save_folder + 'probe0/sorters/kilosort4_merged/')
merge_suggestions = sio.loadmat(analysis_folder + 'probe1um_merge_suggestion_ks4.mat')
match_ids = merge_suggestions['match_ids']
merge_ids = match_ids[:,1]
original_ids = match_ids[:,0]
cs_probe1 = si.CurationSorting(parent_sorting=probe1_sorting_ks4)
unique_ids = np.unique(merge_ids)
for id in unique_ids:
    id_count = np.count_nonzero(merge_ids == id)
    if id_count > 1:
        units = np.where(merge_ids == id)
        units_index = (units[0]-1,)
        cs_probe1.merge(original_ids[units_index])

probe1_sorting_ks4_merged = cs_probe1.sorting
probe1_sorting_ks4_merged.save(folder = save_folder + 'probe1/sorters/kilosort4_merged/')

merge_suggestions = sio.loadmat(analysis_folder + 'probe0um_merge_suggestion_ks3.mat')
match_ids = merge_suggestions['match_ids']
merge_ids = match_ids[:,1]
original_ids = match_ids[:,0]
cs_probe0 = si.CurationSorting(parent_sorting=probe0_sorting_ks3)
unique_ids = np.unique(merge_ids)
for id in unique_ids:
    id_count = np.count_nonzero(merge_ids == id)
    if id_count > 1:
        units = np.where(merge_ids == id)
        units_index = (units[0]-1,)
        cs_probe0.merge(original_ids[units_index])
        
probe0_sorting_ks3_merged = cs_probe0.sorting
probe0_sorting_ks3_merged.save(folder = save_folder + 'probe0/sorters/kilosort3_merged/')
merge_suggestions = sio.loadmat(analysis_folder + 'probe1um_merge_suggestion_ks3.mat')
match_ids = merge_suggestions['match_ids']
merge_ids = match_ids[:,1]
original_ids = match_ids[:,0]
cs_probe1 = si.CurationSorting(parent_sorting=probe1_sorting_ks3)
unique_ids = np.unique(merge_ids)
for id in unique_ids:
    id_count = np.count_nonzero(merge_ids == id)
    if id_count > 1:
        units = np.where(merge_ids == id)
        units_index = (units[0]-1,)
        cs_probe1.merge(original_ids[units_index])

probe1_sorting_ks3_merged = cs_probe1.sorting
probe1_sorting_ks3_merged.save(folder = save_folder + 'probe1/sorters/kilosort3_merged/')

''' Compute quality metrics on the extracted waveforms'''
probe0_we_ks4_merged = si.create_sorting_analyzer(probe0_preprocessed_corrected, probe0_sorting_ks4_merged, folder=save_folder +'probe0/waveform/kilosort4_merged',
                        sparse=True, max_spikes_per_unit=500, ms_before=1.5,ms_after=2.,
                        **job_kwargs)

probe1_we_ks4_merged = si.create_sorting_analyzer(probe1_preprocessed_corrected, probe1_sorting_ks4_merged, folder=save_folder +'probe1/waveform/kilosort4_merged',
                        sparse=True, max_spikes_per_unit=500, ms_before=1.5,ms_after=2.,
                        **job_kwargs)

probe0_we_ks3_merged = si.create_sorting_analyzer(probe0_preprocessed_corrected, probe0_sorting_ks3_merged, folder=save_folder +'probe0/waveform/kilosort3_merged',
                        sparse=True, max_spikes_per_unit=500, ms_before=1.5,ms_after=2.,
                        **job_kwargs)

probe1_we_ks3_merged = si.create_sorting_analyzer(probe1_preprocessed_corrected, probe1_sorting_ks3_merged, folder=save_folder +'probe1/waveform/kilosort3_merged',
                        sparse=True, max_spikes_per_unit=500, ms_before=1.5,ms_after=2.,
                        **job_kwargs)
probe0_we_ks4_merged.compute('waveforms')
probe1_we_ks4_merged.compute('waveforms')
probe0_we_ks3_merged.compute('waveforms')
probe1_we_ks3_merged.compute('waveforms')

probe0_we_ks4_merged.compute('template_metrics')
probe1_we_ks4_merged.compute('template_metrics')
probe0_we_ks3_merged.compute('template_metrics')
probe1_we_ks3_merged.compute('template_metrics')

probe0_we_ks4_merged.compute('noise_levels')
probe1_we_ks4_merged.compute('noise_levels')
probe0_we_ks3_merged.compute('noise_levels')
probe1_we_ks3_merged.compute('noise_levels')

probe0_we_ks4_merged.compute('principal_components',**job_kwargs)
probe1_we_ks4_merged.compute('principal_components',**job_kwargs)
probe0_we_ks3_merged.compute('principal_components',**job_kwargs)
probe1_we_ks3_merged.compute('principal_components',**job_kwargs)

probe0_we_ks4_merged.compute('template_similarity')
probe1_we_ks4_merged.compute('template_similarity')
probe0_we_ks3_merged.compute('template_similarity')
probe1_we_ks3_merged.compute('template_similarity')

probe0_we_ks4_merged.compute('correlograms')
probe1_we_ks4_merged.compute('correlograms')
probe0_we_ks3_merged.compute('correlograms')
probe1_we_ks3_merged.compute('correlograms')

probe0_we_ks4_merged.compute('spike_amplitudes',**job_kwargs)
probe1_we_ks4_merged.compute('spike_amplitudes',**job_kwargs)
probe0_we_ks3_merged.compute('spike_amplitudes',**job_kwargs)
probe1_we_ks3_merged.compute('spike_amplitudes',**job_kwargs)

probe0_we_ks4_merged.compute('isi_histograms')
probe1_we_ks4_merged.compute('isi_histograms')
probe0_we_ks3_merged.compute('isi_histograms')
probe1_we_ks3_merged.compute('isi_histograms')

qm_list = si.get_default_qm_params()
print('The following quality metrics are computed:')
print(qm_list)
probe0_we_ks4_merged.compute('quality_metrics', qm_params=qm_list,**job_kwargs)
probe1_we_ks4_merged.compute('quality_metrics', qm_params=qm_list,**job_kwargs)
probe0_we_ks3_merged.compute('quality_metrics', qm_params=qm_list,**job_kwargs)
probe1_we_ks3_merged.compute('quality_metrics', qm_params=qm_list,**job_kwargs)


'''minor corrections to the folder path of files before moving the files to server'''
#process to change all the folder paths in text and .json files on Beast to the server before uploading it to the server
file_list_ks4 =[save_folder + "probe0_preprocessed/provenance.json",
            save_folder + "probe1_preprocessed/provenance.json",
            save_folder + "probe0/waveform/kilosort4/recording.json",
            save_folder + "probe0/waveform/kilosort4/sorting.json",
            save_folder + 'probe0/sorters/kilosort4/spikeinterface_recording.json',
            save_folder + "probe0/sorters/kilosort4/in_container_sorting/provenance.json",
            save_folder + "probe0/sorters/kilosort4/in_container_sorting/si_folder.json",
            save_folder + 'probe0/waveform/kilosort4_merged/recording.json',
            save_folder + 'probe0/waveform/kilosort4_merged/sorting.json',
            save_folder + 'probe0/sorters/kilosort4_merged/spikeinterface_recording.json',
            save_folder + 'probe0/sorters/kilosort4_merged/in_container_sorting/provenance.json',
            save_folder + 'probe0/sorters/kilosort4_merged/in_container_sorting/si_folder.json',
            save_folder + "probe1/waveform/kilosort4/recording.json",
            save_folder + "probe1/waveform/kilosort4/sorting.json",
            save_folder + 'probe1/sorters/kilosort4/spikeinterface_recording.json',
            save_folder + "probe1/sorters/kilosort4/in_container_sorting/provenance.json",
            save_folder + "probe1/sorters/kilosort4/in_container_sorting/si_folder.json",
            save_folder + 'probe0/waveform/kilosort4_merged/recording.json',
            save_folder + 'probe0/waveform/kilosort4_merged/sorting.json',
            save_folder + 'probe0/sorters/kilosort4_merged/spikeinterface_recording.json',
            save_folder + 'probe0/sorters/kilosort4_merged/in_container_sorting/provenance.json',
            save_folder + 'probe0/sorters/kilosort4_merged/in_container_sorting/si_folder.json']
file_list_ks3 =[
            save_folder + "probe0/waveform/kilosort3/recording.json",
            save_folder + "probe0/waveform/kilosort3/sorting.json",
            save_folder + 'probe0/sorters/kilosort3/spikeinterface_recording.json',
            save_folder + "probe0/sorters/kilosort3/in_container_sorting/provenance.json",
            save_folder + "probe0/sorters/kilosort3/in_container_sorting/si_folder.json",
            save_folder + 'probe0/waveform/kilosort3_merged/recording.json',
            save_folder + 'probe0/waveform/kilosort3_merged/sorting.json',
            save_folder + 'probe0/sorters/kilosort3_merged/spikeinterface_recording.json',
            save_folder + 'probe0/sorters/kilosort3_merged/in_container_sorting/provenance.json',
            save_folder + 'probe0/sorters/kilosort3_merged/in_container_sorting/si_folder.json',
            save_folder + "probe1/waveform/kilosort3/recording.json",
            save_folder + "probe1/waveform/kilosort3/sorting.json",
            save_folder + 'probe1/sorters/kilosort3/spikeinterface_recording.json',
            save_folder + "probe1/sorters/kilosort3/in_container_sorting/provenance.json",
            save_folder + "probe1/sorters/kilosort3/in_container_sorting/si_folder.json",
            save_folder + 'probe0/waveform/kilosort3_merged/recording.json',
            save_folder + 'probe0/waveform/kilosort3_merged/sorting.json',
            save_folder + 'probe0/sorters/kilosort3_merged/spikeinterface_recording.json',
            save_folder + 'probe0/sorters/kilosort3_merged/in_container_sorting/provenance.json',
            save_folder + 'probe0/sorters/kilosort3_merged/in_container_sorting/si_folder.json']
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

for files in file_list_ks4:
    
    # open the JSON file and load the data
    with open(files, 'r') as f:
        data = json.load(f)
    
    # replace the text
    data = replace_text(data, save_folder[:-1], ephys_folder[:-1])
    
    # write the updated data back to the JSON file
    with open(files, 'w') as f:
        json.dump(data, f, indent=4)
        
for files in file_list_ks3:
    
    # open the JSON file and load the data
    with open(files, 'r') as f:
        data = json.load(f)
    
    # replace the text
    data = replace_text(data, save_folder[:-1], ephys_folder[:-1])
    
    # write the updated data back to the JSON file
    with open(files, 'w') as f:
        json.dump(data, f, indent=4)

# #delete temp_wh.dat files
# dat_files = [ dst_folder + "probe0/sorters/kilosort2_5/sorter_output/temp_wh.dat",
#             dst_folder + "probe0/sorters/kilosort3/sorter_output/temp_wh.dat",
#             dst_folder + "probe1/sorters/kilosort2_5/sorter_output/temp_wh.dat",
#             dst_folder + "probe1/sorters/kilosort3/sorter_output/temp_wh.dat"]
# for files in dat_files:
#     os.remove(files)
#move spikeinterface folder on Beast to the server

import shutil
import os

folders_to_move = ['probe0',
                'probe1',
                'probe0_preprocessed',
                'probe1_preprocessed']
##
#
for folder in folders_to_move:
    # construct the destination path
    destination = os.path.join(base_folder + mouse + '/ephys/' +save_folder, folder)
    # copy the folder to the destination
    shutil.copytree(save_folder+folder, destination)
#
#remove all temmp files
shutil.rmtree(save_folder)

print('All Done! Overall it took:')

print(datetime.now() - startTime)
print('to finish! Please move the files to the server as soon as you have time!')
