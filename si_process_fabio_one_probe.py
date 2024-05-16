def si_process_fabio_one_probe(base_folder, mouse, date,dst_folder,job_kwargs):
    # get all the recordings on that day
    ephys_folder = base_folder + mouse + '/ephys/' + date +'/'
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
    import docker


    from datetime import datetime
    
    startTime = datetime.now()
    print('Start Time:' + startTime.strftime("%m/%d/%Y, %H:%M:%S"))
    ''' this section defines the animal and dates and fetch the recordings from the server to Beast'''
    print('copying ephys data from:' + ephys_folder)
    g_files = []
    # iterate over all directories in source folder
    
    for dirname in os.listdir(ephys_folder):
        # check if '_g' is in the directory name
        #only grab recording folders - there might be some other existing folders for analysis or sorted data
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
    def sorting_key(s):
        return int(s.split('_g')[-1])

    # Sort the list using the custom sorting key
    g_files = sorted(g_files, key=sorting_key)

    print(g_files) 
    stream_names, stream_ids = si.get_neo_streams('spikeglx',dst_folder)
    print(stream_names)
    print(stream_ids)
    #load first probe from beast folder - MEC probe for Diao
    probe0_raw = si.read_spikeglx(dst_folder,stream_name='imec0.ap')
    print(probe0_raw)
    
    import itertools
    probe0_num_segments = [probe0_raw.get_num_frames(segment_index=i) for i in range(probe0_raw.get_num_segments())]
    probe0_end_sample_frames = list(itertools.accumulate(probe0_num_segments))
    probe0_start_sample_frames = [1] + [probe0_end_sample_frames[i] + 1 for i in range(0, len(probe0_num_segments)-1)]
   
    #several preprocessing steps and concatenation of the recordings
    #highpass filter - threhsold at 300Hz
    probe0_highpass = si.highpass_filter(probe0_raw,freq_min=300.)
    #detect bad channels
    bad_channel_ids, channel_labels = si.detect_bad_channels(probe0_highpass)
    #remove bad channels if wanted
    probe0_remove_channels = probe0_highpass.remove_channels(bad_channel_ids)
    print('probe0_bad_channel_ids',bad_channel_ids)
    #phase shift correction - equivalent to T-SHIFT in catGT
    probe0_phase_shift = si.phase_shift(probe0_remove_channels)
    probe0_common_reference = si.common_reference(probe0_phase_shift,operator='median',reference='global')
    probe0_preprocessed = probe0_common_reference
    probe0_cat = si.concatenate_recordings([probe0_preprocessed])
    print('probe0_preprocessed',probe0_preprocessed)
    print('probe0 concatenated',probe0_cat)

    
    '''Motion Drift Correction'''
    #motion correction if needed
    #this is nonrigid correction - need to do parallel computing to speed up
    #assign parallel processing as job_kwargs
    
    probe0_nonrigid_accurate = si.correct_motion(recording=probe0_cat, preset="nonrigid_accurate",**job_kwargs)

    print('Start to motion correction finished:')
    print(datetime.now() - startTime)
    #kilosort like to mimic kilosort - no need if you are just running kilosort
    #probe0_kilosort_like = correct_motion(recording=probe0_cat, preset="kilosort_like")
    #probe1_kilosort_like = correct_motion(recording=probe1_cat, preset="kilosort_like")

    '''save preprocessed bin file before sorting'''


    #after saving, sorters will read this preprocessed binary file instead
    probe0_preprocessed_corrected = probe0_nonrigid_accurate.save(folder=dst_folder+'probe0_preprocessed', format='binary', **job_kwargs)
    print('Start to prepocessed files saved:')
    print(datetime.now() - startTime)
    #probe0_preprocessed_corrected = si.load_extractor(dst_folder+'/probe0_preprocessed')
    #probe1_preprocessed_corrected = si.load_extractor(dst_folder+'/probe1_preprocessed')
    ''' prepare sorters - currently using the default parameters and motion correction is turned off as it was corrected already above
        you can check if the parameters using:
        params = get_default_sorter_params('kilosort3')
    print("Parameters:\n", params)

    desc = get_sorter_params_description('kilosort3')
    print("Descriptions:\n", desc)

    Beware that moutainsort5 is commented out as the sorter somehow stops midway with no clue - currently raising this issue on their github page
    '''
    probe0_sorting_ks4 = si.run_sorter(sorter_name= 'kilosort4',recording=probe0_preprocessed_corrected,output_folder=dst_folder+'probe0/sorters/kilosort4/',docker_image=True,do_correction=False)
    #job_list = [
    # {'sorter_name':'kilosort2_5','recording':probe0_preprocessed_corrected,'output_folder':dst_folder+'/probe0/sorters/kilosort2_5/','docker_image':'spikeinterface/kilosort2_5-compiled-base','do_correction':False},
    #  {'sorter_name':'kilosort3','recording':probe0_preprocessed_corrected,'output_folder':dst_folder+'/probe0/sorters/kilosort3/','docker_image':True,'do_correction':False},
    # {'sorter_name':'mountainsort5','recording':probe0_preprocessed_corrected,'output_folder':dst_folder+'/probe0/sorters/mountainsort5/','docker_image':True},
    #    {'sorter_name':'kilosort2_5','recording':probe1_preprocessed_corrected,'output_folder':dst_folder+'/probe1/sorters/kilosort2_5/','docker_image':'spikeinterface/kilosort2_5-compiled-base','do_correction':False},
    #  {'sorter_name':'kilosort3','recording':probe1_preprocessed_corrected,'output_folder':dst_folder+'/probe1/sorters/kilosort3/','docker_image':True,'do_correction':False},
    # {'sorter_name':'mountainsort5','recording':probe1_preprocessed_corrected,'output_folder':dst_folder+'/probe1/sorters/mountainsort5/','docker_image':True},
    #]
    #run sorters in parallel
    #sortings = si.run_sorter_jobs(job_list = job_list,engine = 'joblib',engine_kwargs = {'n_jobs': 2})
    #remove duplicates
    probe0_sorting_ks4 = si.remove_duplicated_spikes(sorting = probe0_sorting_ks4, censored_period_ms=0.3,method='keep_first')
    print(probe0_sorting_ks4)
    print('Start to all sorting done:')
    print(datetime.now() - startTime)

    import pandas as pd
    probe0_segment_frames = pd.DataFrame({'segment_info':g_files,'segment start frame': probe0_start_sample_frames, 'segment end frame': probe0_end_sample_frames})
    probe0_segment_frames.to_csv(dst_folder+'probe0/sorters/segment_frames.csv', index=False)
   


    ''' read sorters directly from the output folder - so you dont need to worry if something went wrong and you can't access the temp variables
        This section reads sorter outputs and extract waveforms 
    '''
    #extract waveforms from sorted data
   

    #probe1_sorting_ks2_5 = spikeinterface.sorters.read_sorter_folder(dst_folder+'/probe1/sorters/kilosort2_5/', register_recording=True, sorting_info=True, raise_error=True)
    probe0_we_ks4 = si.extract_waveforms(probe0_preprocessed_corrected, probe0_sorting_ks4, folder=dst_folder +'probe0/waveform/kilosort4',
                            sparse=True, max_spikes_per_unit=500, ms_before=1.5,ms_after=2.,
                            **job_kwargs)
   
    ''' Compute quality metrics on the extracted waveforms'''
    template_metric_probe0_ks4 = si.compute_template_metrics(probe0_we_ks4)

    noise_levels_probe0_ks4 = si.compute_noise_levels(probe0_we_ks4)

    PCA_probe0_ks4 = si.compute_principal_components(probe0_we_ks4,**job_kwargs)


    template_similarity_probe0_ks4 = si.compute_template_similarity(probe0_we_ks4)

    correlograms_probe0_ks4 = si.compute_correlograms(probe0_we_ks4)

    amplitudes_probe0_ks4 = si.compute_spike_amplitudes(probe0_we_ks4,**job_kwargs)
    

    isi_histograms_probe0_ks4 = si.compute_isi_histograms(probe0_we_ks4)

    qm_list = si.get_quality_metric_list()
    print('The following quality metrics are computed:')
    print(qm_list)
    probe0_ks4_metrics = si.compute_quality_metrics(probe0_we_ks4, metric_names=qm_list,**job_kwargs)

    '''minor corrections to the folder path of files before moving the files to server'''
    #process to change all the folder paths in text and .json files on Beast to the server before uploading it to the server
    file_list = [dst_folder + "probe0_preprocessed/provenance.json",
                dst_folder + "probe0/waveform/kilosort4/recording.json",
                dst_folder + "probe0/waveform/kilosort4/sorting.json",
                dst_folder + "probe0/sorters/kilosort4/in_container_sorting/provenance.json",
                dst_folder + "probe0/sorters/kilosort4/in_container_sorting/si_folder.json"]

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

    for files in file_list:
        
        # open the JSON file and load the data
        with open(files, 'r') as f:
            data = json.load(f)
        
        # replace the text
        data = replace_text(data, dst_folder[:-1], ephys_folder[:-1])
        
        # write the updated data back to the JSON file
        with open(files, 'w') as f:
            json.dump(data, f, indent=4)

    #move spikeinterface folder on Beast to the server

    import shutil
    import os

    folders_to_move = ['probe0',
                    'probe0_preprocessed']
##
#
    for folder in folders_to_move:
        # construct the destination path
        destination = os.path.join(ephys_folder, folder)
        # copy the folder to the destination
        shutil.copytree(dst_folder+folder, destination)
#
    #remove all temmp files
#    shutil.rmtree(dst_folder)

    print('All Done! Overall it took:')

    print(datetime.now() - startTime)
    print('to finish! Please move the files to the server as soon as you have time!')
