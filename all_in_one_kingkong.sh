#!/bin/bash

# Define variables
mouse='M24064' #mouse id
save_date='20241213' #date of recording
dates='20241213/20241213_0' #acquisition date and session e.g. dates='20240624/20240624_0,20240624/20240624_1'
base_folder='/home/masa/spikesorting_temp_data/'  # local folder of godzilla
no_probe=2 #number of probes you have in this session
use_ks4=true #use kilosort4 
use_ks3=true #use kilosort3
server_folder='/home/masa/rds01/ibn-vision/DATA/SUBJECTS/' #server folder where the data is stored
g_files_to_ignore='[[],[0,1],[0,1]]' #files to ignore for each probe

# Run the first Python script with inputs
python download_zarr_kingkong.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
python premerging_zarr_kingkong_probe0.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
python premerging_zarr_kingkong_probe1.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
#python premerging_zarr_kingkong.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder $g_files_to_ignore
#python premerging_zarr.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder $g_files_to_ignore

# Run the MATLAB script for unit_match_merge_ks4_one_probe
matlab -nodisplay -nosplash -r "mouse='${mouse}'; date='${save_date}'; base_folder='${base_folder}';noprobe='${no_probe}'; run('unit_match_merge_ks4.m'); exit;"
matlab -nodisplay -nosplash -r "mouse='${mouse}'; date='${save_date}'; base_folder='${base_folder}';noprobe='${no_probe}'; run('unit_match_merge_ks3.m'); exit;"
# Run the MATLAB script for the ks3 version (assuming the script name and required adjustments)
# matlab -nodisplay -nosplash -r "mouse='$mouse'; date=''; base_folder=''; run('your_ks3_script_name.m'); exit;"

# Run the second Python script with inputs
python merging_kingkong_probe1.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3
python merging_kingkong_probe0.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3
