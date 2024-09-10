#!/bin/bash

# Define variables
mouse='M24017' #mouse id
save_date='20240606' #date of recording
dates='20240606/20240606_0' #acquisition date and session e.g. dates='20240624/20240624_0,20240624/20240624_1'
base_folder='/home/saleem_lab/spikeinterface_sorting/temp_data/'  # local folder of godzilla
no_probe=2 #number of probes you have in this session
use_ks4=false #use kilosort4
use_ks3=true #use kilosort3
export KILOSORT3_PATH=/home/saleem_lab/Kilosort #path to kilosort3
# g_files_to_ignore input currently not working for some reasons.... but it works if it is inside
g_files_to_ignore=['tcat'] # e.g. g_files_to_ignore=['tcat','0_g6','0_g7','0_g8','0_g9']
# Run the first Python script with inputs
#python premerging_masa.py $mouse $dates $save_date $base_folder $no_probe $use_ks4 $use_ks3 $g_files_to_ignore

#python premerging_masa_temp_godzilla.py $mouse $dates $save_date $base_folder $no_probe $use_ks4 $use_ks3 $g_files_to_ignore
python pre_merging_fix_ks3.py $mouse $dates $save_date $base_folder $no_probe $use_ks4 $use_ks3 $g_files_to_ignore

#python premerging.py $mouse $dates $save_date $base_folder $no_probe $use_ks4 $use_ks3
# Run the MATLAB script for unit_match_merge_ks4_one_probe
#matlab -nodisplay -nosplash -r "mouse='${mouse}'; date='${save_date}'; base_folder='${base_folder}';noprobe='${no_probe}'; run('unit_match_merge_ks4.m'); exit;"
matlab -nodisplay -nosplash -r "mouse='${mouse}'; date='${save_date}'; base_folder='${base_folder}';noprobe='${no_probe}'; run('unit_match_merge_ks3.m'); exit;"
# Run the MATLAB script for the ks3 version (assuming the script name and required adjustments)
# matlab -nodisplay -nosplash -r "mouse='$mouse'; date=''; base_folder=''; run('your_ks3_script_name.m'); exit;"
use_ks4=false #use kilosort4
use_ks3=true #use kilosort3
# Run the second Python script with inputs
python merging_temp2.py $mouse $dates $save_date $base_folder $no_probe $use_ks4 $use_ks3