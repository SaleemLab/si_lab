
from si_process import si_process
from si_process_one_probe import si_process_one_probe
import os
import subprocess
subprocess.run('ulimit -n 4096',shell=True)
#grab recordings from the server to local machine (Beast)
base_folder = '/mnt/rds01/ibn-vision/DATA/SUBJECTS/'

job_kwargs = dict(n_jobs=32, chunk_duration='1s', progress_bar=True)
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
#mouse = 'M23032'
#dates = ['20230719','20230720','20230721','20230722']
mouse = 'M23034'
dates = ['20230807']
for date in dates:
    
    # get all the recordings on that day
    ephys_folder = base_folder + mouse + '/ephys/' + date +'/'
    dst_folder = "/home/saleem_lab/spikeinterface_sorting/temp_data/" + date + '/'
    startTime = datetime.now()
    print('Start Time:' + startTime.strftime("%m/%d/%Y, %H:%M:%S"))
    ''' this section defines the animal and dates and fetch the recordings from the server to Beast'''
    print('copying ephys data from:' + ephys_folder)

    # iterate over all directories in source folder


    folders_to_move = ['probe0_preprocessed', 'probe1_preprocessed']
    for folder in folders_to_move:
        # construct the destination path
        destination = os.path.join(dst_folder, folder)
        # copy the folder to the destination
        shutil.copytree(ephys_folder+folder, destination)
    print('Start to copying files to Beast:')
    print(datetime.now() - startTime)
    ''' read spikeglx recordings and preprocess them'''
    # Define a custom sorting key that extracts the number after 'g'

    #load first probe from beast folder - MEC probe for Diao
    probe0_preprocessed_corrected = si.load_extractor(dst_folder + 'probe0_preprocessed')
    probe1_preprocessed_corrected = si.load_extractor(dst_folder + 'probe1_preprocessed')
    print(probe0_preprocessed_corrected)


    probe0_sorting_ks4 = si.run_sorter(sorter_name= 'kilosort4',recording=probe0_preprocessed_corrected,output_folder=dst_folder+'probe0/sorters/kilosort4/',docker_image=True,do_correction=False)
    probe1_sorting_ks4 = si.run_sorter(sorter_name= 'kilosort4',recording=probe1_preprocessed_corrected,output_folder=dst_folder+'probe1/sorters/kilosort4/',docker_image="spikeinterface/kilosort4-base:latest",do_correction=False)

    #run sorters in parallel
    #sortings = si.run_sorter_jobs(job_list = job_list,engine = 'joblib',engine_kwargs = {'n_jobs': 2})
    #remove duplicates
    probe0_sorting_ks4 = si.remove_duplicated_spikes(sorting = probe0_sorting_ks4, censored_period_ms=0.3,method='keep_first')
    probe1_sorting_ks4 = si.remove_duplicated_spikes(sorting = probe1_sorting_ks4, censored_period_ms=0.3,method='keep_first')
    #probe0_sorting_ks4 = si.remove_excess_spikes(probe0_sorting_ks4 , probe0_preprocessed_corrected)
    #probe1_sorting_ks4 = si.remove_excess_spikes(probe1_sorting_ks4 , probe1_preprocessed_corrected)
    print(probe0_sorting_ks4)
    print(probe1_sorting_ks4)
    print('Start to all sorting done:')
    print(datetime.now() - startTime)

    ''' read sorters directly from the output folder - so you dont need to worry if something went wrong and you can't access the temp variables
        This section reads sorter outputs and extract waveforms 
    '''
    #extract waveforms from sorted data

    #probe0_sorting_ks4 = spikeinterface.sorters.read_sorter_folder(dst_folder+'/probe0/sorters/kilosort2_5/', register_recording=True, sorting_info=True, raise_error=True)
    probe0_we_ks4 = si.extract_waveforms(probe0_preprocessed_corrected, probe0_sorting_ks4, folder=dst_folder +'probe0/waveform/kilosort4',
                            sparse=True, max_spikes_per_unit=500, ms_before=1.5,ms_after=2.,
                            **job_kwargs)

    #probe0_sorting_ks4 = spikeinterface.sorters.read_sorter_folder(dst_folder+'/probe0/sorters/kilosort3/', register_recording=True, sorting_info=True, raise_error=True)
    probe1_we_ks4 = si.extract_waveforms(probe1_preprocessed_corrected, probe1_sorting_ks4, folder=dst_folder +'probe1/waveform/kilosort4',
                            sparse=True, max_spikes_per_unit=500, ms_before=1.5,ms_after=2.,
                            **job_kwargs)

    ''' Compute quality metrics on the extracted waveforms'''
    template_metric_probe0_ks4 = si.compute_template_metrics(probe0_we_ks4)
    template_metric_probe1_ks4 = si.compute_template_metrics(probe1_we_ks4)
    noise_levels_probe0_ks4 = si.compute_noise_levels(probe0_we_ks4)
    noise_levels_probe1_ks4 = si.compute_noise_levels(probe1_we_ks4)
    PCA_probe0_ks4 = si.compute_principal_components(probe0_we_ks4,**job_kwargs)
    PCA_probe1_ks4 = si.compute_principal_components(probe1_we_ks4,**job_kwargs)
    template_similarity_probe0_ks4 = si.compute_template_similarity(probe0_we_ks4)
    template_similarity_probe1_ks4 = si.compute_template_similarity(probe1_we_ks4)
    correlograms_probe0_ks4 = si.compute_correlograms(probe0_we_ks4)
    correlograms_probe1_ks4 = si.compute_correlograms(probe1_we_ks4)
    amplitudes_probe0_ks4 = si.compute_spike_amplitudes(probe0_we_ks4,**job_kwargs)
    amplitudes_probe1_ks4 = si.compute_spike_amplitudes(probe1_we_ks4,**job_kwargs)
    isi_histograms_probe0_ks4 = si.compute_isi_histograms(probe0_we_ks4)
    isi_histograms_probe1_ks4 = si.compute_isi_histograms(probe1_we_ks4)
    qm_list = si.get_quality_metric_list()
    print('The following quality metrics are computed:')
    print(qm_list)
    probe0_ks4_metrics = si.compute_quality_metrics(probe0_we_ks4, metric_names=qm_list,**job_kwargs)
    probe1_ks4_metrics = si.compute_quality_metrics(probe1_we_ks4, metric_names=qm_list,**job_kwargs)
    '''minor corrections to the folder path of files before moving the files to server'''
    #process to change all the folder paths in text and .json files on Beast to the server before uploading it to the server
    file_list = [
                dst_folder + "probe0/waveform/kilosort4/recording.json",
                dst_folder + "probe0/waveform/kilosort4/sorting.json",
                dst_folder + "probe1/waveform/kilosort4/recording.json",
                dst_folder + "probe1/waveform/kilosort4/sorting.json",
                dst_folder + "probe0/sorters/kilosort4/in_container_sorting/provenance.json",
                dst_folder + "probe0/sorters/kilosort4/in_container_sorting/si_folder.json",
                dst_folder + "probe1/sorters/kilosort4/in_container_sorting/provenance.json",
                dst_folder + "probe1/sorters/kilosort4/in_container_sorting/si_folder.json"]
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
    #delete temp_wh.dat files
    #move spikeinterface folder on Beast to the server
    import shutil
    import os
    folders_to_move = ['probe0/waveform/kilosort4',
                    'probe0/sorters/kilosort4',
                    'probe1/waveform/kilosort4',
                    'probe1/sorters/kilosort4']
    for folder in folders_to_move:
        # construct the destination path
        destination = os.path.join(ephys_folder, folder)
        # copy the folder to the destination
        shutil.copytree(dst_folder+folder, destination)
    #remove all temmp files
    shutil.rmtree(dst_folder)
    print('All Done! Overall it took:')
    print(datetime.now() - startTime)
    print('to finish')