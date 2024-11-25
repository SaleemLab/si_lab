# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
def save_spikes_to_csv(spikes,save_folder):
    unit_index = spikes['unit_index']
    segment_index = spikes['segment_index']
    sample_index = spikes['sample_index']
    spikes_df = pd.DataFrame({'unit_index':unit_index,'segment_index':segment_index,'sample_index':sample_index})
    spikes_df.to_csv(save_folder + 'spikes.csv',index=False)
    
base_folder = '//rdp.arc.ucl.ac.uk/ritd-ag-project-rd01ie-asale69/ibn-vision/DATA/SUBJECTS/' 
mouse = 'M24017'
dates = ['20240604','20240606','20240607','20240613']
#dates = ['20230711','20230712','20230713','20230714']

# For non-merged clusters
for date in dates:
    save_folder = base_folder + '/' + mouse + '/ephys/' + date + '/'
    for probe in [0,1]:
        probe0_ks3_spikes = np.load(save_folder + 'probe'+str(probe)+'/waveform/kilosort3/sorting/spikes.npy')
        save_spikes_to_csv(probe0_ks3_spikes,save_folder + 'probe'+str(probe)+'/waveform/kilosort3/sorting/')
        #probe0_ks4_spikes = np.load(save_folder + 'probe'+str(probe)+'/waveform/kilosort4/sorting/spikes.npy')
        #save_spikes_to_csv(probe0_ks4_spikes,save_folder + 'probe'+str(probe)+'/waveform/kilosort4/sorting/') 
        
mouse = 'M24016'
dates = ['20240626','20240701','20240706']
#dates = ['20230711','20230712','20230713','20230714']

# For non-merged clusters
for date in dates:
    save_folder = base_folder + '/' + mouse + '/ephys/' + date + '/'
    for probe in [0,1]:
        probe0_ks3_spikes = np.load(save_folder + 'probe'+str(probe)+'/waveform/kilosort3/sorting/spikes.npy')
        save_spikes_to_csv(probe0_ks3_spikes,save_folder + 'probe'+str(probe)+'/waveform/kilosort3/sorting/')
        #probe0_ks4_spikes = np.load(save_folder + 'probe'+str(probe)+'/waveform/kilosort4/sorting/spikes.npy')
        #save_spikes_to_csv(probe0_ks4_spikes,save_folder + 'probe'+str(probe)+'/waveform/kilosort4/sorting/')       
        
mouse = 'M24018'
dates = ['20240715','20240718','20240723']
#dates = ['20230711','20230712','20230713','20230714']

# For non-merged clusters
for date in dates:
    save_folder = base_folder + '/' + mouse + '/ephys/' + date + '/'
    for probe in [0,1]:
        probe0_ks3_spikes = np.load(save_folder + 'probe'+str(probe)+'/waveform/kilosort3/sorting/spikes.npy')
        save_spikes_to_csv(probe0_ks3_spikes,save_folder + 'probe'+str(probe)+'/waveform/kilosort3/sorting/')
        #probe0_ks4_spikes = np.load(save_folder + 'probe'+str(probe)+'/waveform/kilosort4/sorting/spikes.npy')
        #save_spikes_to_csv(probe0_ks4_spikes,save_folder + 'probe'+str(probe)+'/waveform/kilosort4/sorting/')       
        
##################################################################################################################
##############################################################################################################
# For merged clusters
base_folder = '//rdp.arc.ucl.ac.uk/ritd-ag-project-rd01ie-asale69/ibn-vision/DATA/SUBJECTS/'
mouse = 'M23017'
#dates = ['20240606']
dates = ['20230628','20230629','20230630','20230701']

        
base_folder = '//rdp.arc.ucl.ac.uk/ritd-ag-project-rd01ie-asale69/ibn-vision/DATA/SUBJECTS/'
mouse = 'M23028'
#dates = ['20240606']
dates = ['20230704','20230705','20230706']
for date in dates:
    save_folder = base_folder + '/' + mouse + '/ephys/' + date + '/'
    for probe in [0]:
        probe0_ks3_spikes = np.load(save_folder + 'probe'+str(probe)+'/waveform/kilosort3_merged/sorting/spikes.npy')
        save_spikes_to_csv(probe0_ks3_spikes,save_folder + 'probe'+str(probe)+'/waveform/kilosort3_merged/sorting/')   
        #probe0_ks4_spikes = np.load(save_folder + 'probe'+str(probe)+'/waveform/kilosort4/sorting/spikes.npy')
        #save_spikes_to_csv(probe0_ks4_spikes,save_folder + 'probe'+str(probe)+'/waveform/kilosort4/sorting/')  
        
base_folder = '//rdp.arc.ucl.ac.uk/ritd-ag-project-rd01ie-asale69/ibn-vision/DATA/SUBJECTS/'
mouse = 'M23029'
#dates = ['20240606']
dates = ['20230706','20230707']
for date in dates:
    save_folder = base_folder + '/' + mouse + '/ephys/' + date + '/'
    for probe in [0]:
        probe0_ks3_spikes = np.load(save_folder + 'probe'+str(probe)+'/waveform/kilosort3_merged/sorting/spikes.npy')
        save_spikes_to_csv(probe0_ks3_spikes,save_folder + 'probe'+str(probe)+'/waveform/kilosort3_merged/sorting/')   
        #probe0_ks4_spikes = np.load(save_folder + 'probe'+str(probe)+'/waveform/kilosort4/sorting/spikes.npy')
        #save_spikes_to_csv(probe0_ks4_spikes,save_folder + 'probe'+str(probe)+'/waveform/kilosort4/sorting/')      

        
import os
import shutil

# Define the source and destination directories
source_dir = r'\\rdp.arc.ucl.ac.uk\ritd-ag-project-rd01ie-asale69\ibn-vision\DATA\SUBJECTS\M23031\analysis'
destination_dir = r'C:\Users\adam.tong\OneDrive - University College London\data\M23031\analysis'

# Walk through the source directory and find all 'extracted_clusters_ks3.mat' files
for root, dirs, files in os.walk(source_dir):
    for file in files:
        if file == 'extracted_clusters_ks3.mat':
            source_file_path = os.path.join(root, file)
            # Construct the destination file path
            relative_path = os.path.relpath(root, source_dir)
            destination_file_path = os.path.join(destination_dir, relative_path, file)
            
            # Ensure the destination directory exists
            os.makedirs(os.path.dirname(destination_file_path), exist_ok=True)
            
            # Copy and replace the file
            shutil.copy2(source_file_path, destination_file_path)
            print(f"Copied and replaced: {source_file_path} -> {destination_file_path}")

print("All files have been copied and replaced.")
