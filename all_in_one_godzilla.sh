#!/bin/bash

# Define variables
mouse='M24016' #mouse id
save_date='20240709' #date of recording
dates='20240709/20240709_0' #acquisition date and session
base_folder='/home/saleem_lab/spikeinterface_sorting/temp_data/'  # local folder of godzilla
no_probe=2 #number of probes you have in this session

# Run the first Python script with inputs
python premerging.py $mouse $dates $save_date $base_folder $no_probe


# Run the MATLAB script for unit_match_merge_ks4_one_probe
matlab -nodisplay -nosplash -r "mouse='${mouse}'; date='${save_date}'; base_folder='${base_folder}';no_probe='${no_probe}'; run('unit_match_merge_ks4.m'); exit;"
matlab -nodisplay -nosplash -r "mouse='${mouse}'; date='${save_date}'; base_folder='${base_folder}';no_probe='${no_probe}'; run('unit_match_merge_ks3.m'); exit;"
# Run the MATLAB script for the ks3 version (assuming the script name and required adjustments)
# matlab -nodisplay -nosplash -r "mouse='$mouse'; date=''; base_folder=''; run('your_ks3_script_name.m'); exit;"



# Run the second Python script with inputs
python merging.py $mouse $dates $save_date $base_folder $no_probe

