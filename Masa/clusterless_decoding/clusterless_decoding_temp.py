# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 08:45:26 2024

@author: masahiro.takigawa
"""
import spikeinterface.core as sc
import spikeinterface.extractors as se
import spikeinterface.full as si
 

mouse = 'M24016'
save_date = ['20240626']
base_folder = 'Z:/ibn-vision/DATA/SUBJECTS/'
no_probe = 2

for date in save_date:
    save_folder = base_folder+mouse+'/ephys/' +date+'/'
    for probe in range(int(no_probe)):
     # Load your preprocessed recording
     probe0_preprocessed_corrected = si.load_extractor(save_folder+'probe'+str(probe)+'_preprocessed')
    
     # Load your sorting
     probe0_ks3_sorting = si.load_extractor(save_folder+'probe'+str(probe)+'/sorters'+'/kilosort3_merged')
    
     # Get the spike times and labels
     spike_times = probe0_ks3_sorting.get_all_spike_trains()
     unit_ids = probe0_ks3_sorting.get_unit_ids()
    
     # ms_before and ms_after
     ms_before = 30 # 30 samples for 1ms
     ms_after = 60 # 60 samples for 1ms
     mode="memmap"
     spikes = probe0_ks3_sorting.to_spike_vector()
     unit_ids=[0];
    
     probe0_ks3_spikes = np.load(save_folder + 'probe'+str(probe)+'/waveform/kilosort3_merged/sorting/spikes.npy')
     probe0_ks3_spikes = np.load(save_folder + 'probe'+str(probe)+'/waveform/kilosort3_merged/sorting/spikes.npy')
     current_spike=probe0_ks3_spikes[0]
     traces = probe0_preprocessed_corrected.get_traces(start_frame=current_spike[0]-ms_before, end_frame=current_spike[0]+ms_after, segment_index=0, return_scaled=True)
    
     #print(np.size(traces,0))
     #print(np.size(traces,1))
    
     import matplotlib.pyplot as plt
     plt.plot(traces)
     plt.show()
        
        