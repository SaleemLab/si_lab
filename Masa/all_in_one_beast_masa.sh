#!/bin/bash

# Define variables
mouse='M24018' #mouse id
save_date='20240715' #date of recording
dates='20240715/20240715_0' #acquisition date and session e.g. dates='20240624/20240624_0,20240624/20240624_1'
base_folder='/home/lab/spikeinterface_sorting/temp_data/'  # Adjust this path if necessary
no_probe=2 #number of probes you have in this session
use_ks4=true #use kilosort4
use_ks3=true #use kilosort3
# g_files_to_ignore input currently not working for some reasons.... but it works if it is inside
g_files_to_ignore=['tcat'] # e.g. g_files_to_ignore=['tcat','0_g6','0_g7','0_g8','0_g9']
# Run the first Python script with inputs
python premerging_masa_temp.py $mouse $dates $save_date $base_folder $no_probe $use_ks4 $use_ks3 $g_files_to_ignore
python premerging_masa_temp2.py $mouse $dates $save_date $base_folder $no_probe $use_ks4 $use_ks3 $g_files_to_ignore
#python premerging_masa.py $mouse $dates $save_date $base_folder $no_probe $use_ks4 $use_ks3 $g_files_to_ignore

#python premerging.py $mouse $dates $save_date $base_folder $no_probe $use_ks4 $use_ks3
# Run the MATLAB script for unit_match_merge_ks4_one_probe
#matlab -nodisplay -nosplash -r "mouse='${mouse}'; date='${save_date}'; base_folder='${base_folder}';noprobe='${no_probe}'; run('unit_match_merge_ks4.m'); exit;"
matlab -nodisplay -nosplash -r "mouse='${mouse}'; date='${save_date}'; base_folder='${base_folder}';noprobe='${no_probe}'; run('unit_match_merge_ks3.m'); exit;"
# Run the MATLAB script for the ks3 version (assuming the script name and required adjustments)
# matlab -nodisplay -nosplash -r "mouse='$mouse'; date=''; base_folder=''; run('your_ks3_script_name.m'); exit;"

# Run the second Python script with inputs
python merging.py $mouse $dates $save_date $base_folder $no_probe $use_ks4 $use_ks3

