
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
base_folder = sys.argv[7]
save_folder = local_folder + save_date +'/'

# Check g files to ignore are correct (tcat should always be ignored)
# Check if sys.argv[8] is empty
if len(sys.argv) > 8 and sys.argv[8]:
    g_files_to_ignore = ast.literal_eval(sys.argv[8])
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
si.set_global_job_kwargs(**job_kwargs)
g_files_all = []
# iterate over all directories in source folder
acquisition_base_path = base_folder + mouse + '/ephys/' + save_date + '/*' + save_date
acquisition_folders = glob.glob(acquisition_base_path + '_*')
acquisition_list = sorted([int(folder.split('_')[-1]) for folder in acquisition_folders])
date_count = 0
    
probes=[0]
for probe in probes:

    preprocessed_corrected = si.load(save_folder+'probe'+str(probe)+'_preprocessed')

    import pandas as pd
    def save_spikes_to_csv(spikes,save_folder):
        unit_index = spikes['unit_index']
        segment_index = spikes['segment_index']
        sample_index = spikes['sample_index']
        spikes_df = pd.DataFrame({'unit_index':unit_index,'segment_index':segment_index,'sample_index':sample_index})
        spikes_df.to_csv(save_folder + 'spikes.csv',index=False)
    extensions = ['templates', 'template_metrics', 'noise_levels', 'template_similarity', 'correlograms', 'isi_histograms']
    if use_ks4:
        sorting_ks4= se.KiloSortSortingExtractor(folder_path=save_folder+'probe'+str(probe)+'/sorters/kilosort4/sorter_output')
 
        we_ks4 = si.create_sorting_analyzer(sorting_ks4, preprocessed_corrected, 
                                format = 'binary_folder',folder=save_folder +'probe'+str(probe)+'/waveform/kilosort4',
                                sparse = True,overwrite = True,
                                **job_kwargs)
        we_ks4.compute('random_spikes')
        we_ks4.compute('waveforms',ms_before=1.0, ms_after=2.0,**job_kwargs)
        ks4_spikes = np.load(save_folder + 'probe'+str(probe)+'/waveform/kilosort4/sorting/spikes.npy')
        save_spikes_to_csv(ks4_spikes,save_folder + 'probe'+str(probe)+'/waveform/kilosort4/sorting/')
        we_ks4.compute(extensions,**job_kwargs)

        we_ks4.compute('principal_components',n_jobs=1,chunk_duration="1s", progress_bar=True)
        we_ks4.compute('spike_amplitudes',**job_kwargs)
        qm_list = si.get_default_qm_params()
        print('The following quality metrics are computed:')
        print(qm_list)
        we_ks4.compute('quality_metrics', qm_params=qm_list,n_jobs=1,chunk_duration="1s", progress_bar=True)
        si.export_report(sorting_analyzer = we_ks4, output_folder = save_folder + 'probe'+str(probe)+'/waveform/kilosort4_report/',**job_kwargs)
        
    if use_ks3:
        sorting_ks3 = si.run_sorter(sorter_name= 'kilosort3',recording=preprocessed_corrected,output_folder=save_folder+'probe'+str(probe)+'/sorters/kilosort3/',docker_image='spikeinterface/kilosort3-compiled-base:latest',do_correction=False)
        sorting_ks3 = si.remove_duplicated_spikes(sorting = sorting_ks3, censored_period_ms=0.3,method='keep_first')
        sorting_ks3 = spikeinterface.curation.remove_excess_spikes(sorting_ks3, preprocessed_corrected)
        sorting_ks3= se.KiloSortSortingExtractor(folder_path=save_folder+'probe'+str(probe)+'/sorters/kilosort3/sorter_output')

        we_ks3 = si.create_sorting_analyzer(sorting_ks3, preprocessed_corrected, 
                                format = 'binary_folder',folder=save_folder +'probe'+str(probe)+'/waveform/kilosort3',
                                sparse = True,overwrite = True,
                                **job_kwargs)
        we_ks3.compute('random_spikes')
        we_ks3.compute('waveforms',ms_before=1.0, ms_after=2.0,**job_kwargs)
        ks3_spikes = np.load(save_folder + 'probe'+str(probe)+'/sorters/kilosort3/in_container_sorting/spikes.npy')
        save_spikes_to_csv(ks3_spikes,save_folder + 'probe'+str(probe)+'/sorters/kilosort3/in_container_sorting/')
        save_spikes_to_csv(ks3_spikes,save_folder + 'probe'+str(probe)+'/waveform/kilosort4/sorting/')
        we_ks3.compute(extensions,**job_kwargs)
        we_ks3.compute('principal_components',n_jobs=1,chunk_duration="1s", progress_bar=True)
        we_ks3.compute('spike_amplitudes',**job_kwargs)
        qm_list = si.get_default_qm_params()
        print('The following quality metrics are computed:')
        print(qm_list)
        we_ks3.compute('quality_metrics', qm_params=qm_list,n_jobs=1,chunk_duration="1s", progress_bar=True)
        si.export_report(sorting_analyzer = we_ks3, output_folder = save_folder + 'probe'+str(probe)+'/waveform/kilosort3_report/',**job_kwargs)
    print('Start to all sorting done:')
    print(datetime.now() - startTime)





    ''' read sorters directly from the output folder - so you dont need to worry if something went wrong and you can't access the temp variables
        This section reads sorter outputs and extract waveforms 
    '''

sys.exit(0)