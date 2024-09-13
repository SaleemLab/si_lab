#!/bin/bash

# Define variables
mouse='M24017' #mouse id
save_date='20240604' #date of recording
dates='20240604/20240604_0' #acquisition date and session e.g. dates='20240624/20240624_0,20240624/20240624_1'
base_folder='/home/saleem_lab/spikeinterface_sorting/temp_data/'  # local folder of godzilla
no_probe=2 #number of probes you have in this session
use_ks4=false #use kilosort4
use_ks3=true #use kilosort3
export KILOSORT3_PATH=/home/saleem_lab/Kilosort #path to kilosort3
# g_files_to_ignore input currently not working for some reasons.... but it works if it is inside
g_files_to_ignore=['tcat'] # e.g. g_files_to_ignore=['tcat','0_g6','0_g7','0_g8','0_g9']
# Run the first Python script with inputs
#python premerging_masa.py $mouse $dates $save_date $base_folder $no_probe $use_ks4 $use_ks3 $g_files_to_ignore
use_ks4=true #use kilosort4
use_ks3=false #use kilosort3
python premerging_masa_local_temp1.py $mouse $dates $save_date $base_folder $no_probe $use_ks4 $use_ks3 $g_files_to_ignore

use_ks4=true #use kilosort4
use_ks3=true #use kilosort3
python premerging_masa_local_temp2.py $mouse $dates $save_date $base_folder $no_probe $use_ks4 $use_ks3 $g_files_to_ignore
#python pre_merging_fix_ks3.py $mouse $dates $save_date $base_folder $no_probe $use_ks4 $use_ks3 $g_files_to_ignore
