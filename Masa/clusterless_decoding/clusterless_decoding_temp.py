# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 08:45:26 2024

@author: masahiro.takigawa
"""
import spikeinterface.core as sc
import spikeinterface.extractors as se
import spikeinterface.full as si
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.signal import find_peaks

mouse = 'M24016'
save_date = ['20240626']
base_folder = 'Z:/ibn-vision/DATA/SUBJECTS/'
no_probe = 2

for date in save_date:
    save_folder = base_folder+mouse+'/ephys/' +date+'/'
    for probe in range(int(no_probe)):
     # Load your preprocessed recording
     probe0_preprocessed_corrected = si.load_extractor(save_folder+'probe'+str(probe)+'_preprocessed')
     # Load channel locations
     channel_locations = np.load(save_folder+'probe'+str(probe)+'_preprocessed/properties/location.npy')
     
     # K mean clustering to get V1 channels and HPC channels
     print(channel_locations)
     kmeans = KMeans(n_clusters=2)
     kmeans.fit(channel_locations)
     
     HPC_channels = np.copy(kmeans.labels_)
     V1_channels = np.copy(kmeans.labels_)
     
     if np.mean(channel_locations[kmeans.labels_==1,1]) > np.mean(channel_locations[kmeans.labels_==0,1]):
         HPC_channels[kmeans.labels_==0]=1;
         HPC_channels[kmeans.labels_==1]=0;
     else:
         V1_channels[kmeans.labels_==0]=1;
         V1_channels[kmeans.labels_==1]=0;  
         
     HPC_channels = np.where(HPC_channels == 1)[0]
     V1_channels = np.where(V1_channels == 1)[0]
     
     # Find HPC channels that jump to next shank
     sorted_channels = np.argsort(channel_locations[HPC_channels,0])
     jump_channels = np.where(np.abs(np.diff(channel_locations[HPC_channels[sorted_channels],0]))>100)[0]
     jump_channels = jump_channels+1
     jump_channels_1 = np.append(0,jump_channels)
     jump_channels = np.append(jump_channels,len(sorted_channels))
     
     
     channel_clusters = []
     for iChannel in  range(int(len(jump_channels))):
         channel_range = list(range(jump_channels_1[iChannel], jump_channels[iChannel]))
         this_shank_channels = HPC_channels[sorted_channels[channel_range]]
         
         # sort channels from this shank according to y axis
         sorted_channels_y = np.argsort(channel_locations[this_shank_channels,1])
         sort_channel_y_locations = channel_locations[this_shank_channels[sorted_channels_y],1]
         
         jump_channels_y = np.where(abs(np.diff(sort_channel_y_locations)>20))[0]
         if any(jump_channels_y):
             jump_channels_y = jump_channels_y+1
             jump_channels_y_1 = np.append(0,jump_channels_y)
             jump_channels_y = np.append(jump_channels_y,len(sort_channel_y_locations))
             
             for nChannel in  range(int(len(jump_channels_y))):
                 channel_range = list(range(jump_channels_y_1[nChannel], jump_channels_y[nChannel]))
                 # Split into equal groups of 6 channels
                 temp = np.array_split(this_shank_channels[sorted_channels_y[channel_range]], len(channel_range) / 6)
                 
                 channel_clusters = channel_clusters + temp
         else:
             # Split into equal groups of 6 channels
             temp = np.array_split(this_shank_channels[sorted_channels_y], len(this_shank_channels) / 6)
             
             channel_clusters = channel_clusters + temp
         
     # Find V1 channels that jump to next shank
     sorted_channels = np.argsort(channel_locations[V1_channels,0])
     jump_channels = np.where(np.abs(np.diff(channel_locations[V1_channels[sorted_channels],0]))>100)[0]
     jump_channels = jump_channels+1
     jump_channels_1 = np.append(0,jump_channels)
     jump_channels = np.append(jump_channels,len(sorted_channels))
     
     
     channel_clusters_V1 = []
     for iChannel in  range(int(len(jump_channels))):
         channel_range = list(range(jump_channels_1[iChannel], jump_channels[iChannel]))
         this_shank_channels = V1_channels[sorted_channels[channel_range]]
         
         # sort channels from this shank according to y axis
         sorted_channels_y = np.argsort(channel_locations[this_shank_channels,1])
         sort_channel_y_locations = channel_locations[this_shank_channels[sorted_channels_y],1]
         
         jump_channels_y = np.where(abs(np.diff(sort_channel_y_locations)>20))[0]
         if any(jump_channels_y):
             jump_channels_y = jump_channels_y+1
             jump_channels_y_1 = np.append(0,jump_channels_y)
             jump_channels_y = np.append(jump_channels_y,len(sort_channel_y_locations))
             
             for nChannel in  range(int(len(jump_channels_y))):
                 channel_range = list(range(jump_channels_y_1[nChannel], jump_channels_y[nChannel]))
                 # Split into equal groups of 6 channels
                 temp = np.array_split(this_shank_channels[sorted_channels_y[channel_range]], len(channel_range) / 6)
                 
                 channel_clusters_V1 = channel_clusters_V1 + temp
         else:
             # Split into equal groups of 6 channels
             temp = np.array_split(this_shank_channels[sorted_channels_y], len(this_shank_channels) / 6)
             
             channel_clusters_V1 = channel_clusters_V1 + temp
         
         
     for i in range(int(len(channel_clusters_V1))):
         plt.scatter(channel_locations[channel_clusters_V1[i],0],channel_locations[channel_clusters_V1[i],1])
         
     plt.show()

     # Load your sorting
     probe0_ks3_sorting = si.load_extractor(save_folder+'probe'+str(probe)+'/sorters'+'/kilosort3_merged')
    
     # Get the spike times and labels
     #spike_times = probe0_ks3_sorting.get_all_spike_trains()
     unit_ids = probe0_ks3_sorting.get_unit_ids()
    
     # ms_before and ms_after
     ms_before = 30 # 30 samples for 1ms
     ms_after = 60 # 60 samples for 1ms
     mode="memmap"
     spikes = probe0_ks3_sorting.to_spike_vector()
     #unit_ids=[0];
    
     V1_spikes
     HPC_spikes
     
     probe0_ks3_spikes = np.load(save_folder + 'probe'+str(probe)+'/waveform/kilosort3_merged/sorting/spikes.npy')
     for current_spike in probe0_ks3_spikes[2]:
         
         current_spike=probe0_ks3_spikes[0]
         traces = np.abs(probe0_preprocessed_corrected.get_traces(start_frame=current_spike[0]-ms_before, end_frame=current_spike[0]+ms_after, segment_index=0, return_scaled=True))
         spike_amplitudes = np.max(traces[30:90,:],0)-np.mean(traces[0:29,:],0) # peak amplitude- baseline before spikes
         peak_channel = find_peaks(spike_amplitudes/np.max(spike_amplitudes),1)
         peak_channel = peak_channel[0]
         
         # Initialize a list to store the indices of the arrays that contain the value
         indices_with_value = []
        
         # Iterate through the HPC channel groups and check for the peak channel
         for index, arr in enumerate(channel_clusters):
             if peak_channel in arr:
                 indices_with_value.append(index)
        
         if any(indices_with_value):
            indices_with_value = indices_with_value[0]
            spike_amplitudes[channel_clusters[indices_with_value]]
        
         # Iterate through the V1 channel groups and check for the peak channel
         for index, arr in enumerate(channel_clusters_V1):
             if peak_channel in arr:
                 indices_with_value.append(index)
        
         if any(indices_with_value):
             indices_with_value = indices_with_value[0]
             spike_amplitudes[channel_clusters_V1[indices_with_value]]
            
     #print(np.size(traces[0:29,:],0))
     #print(np.size(channel_locations,1))
    
     
     
     plt.plot(spike_amplitudes)
     plt.show()
        
        