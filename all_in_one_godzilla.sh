#!/bin/bash

# Define variables
mouse='M24019'
save_date='20240703'
dates='20240703/20240703_0'
base_folder='/home/saleem_lab/spikeinterface_sorting/temp_data/'  # Adjust this path if necessary

# Navigate to the script directory
cd godzilla

# Run the first Python script with inputs
python godzilla_premerging_one_probe.py $mouse $dates $save_date

# Navigate back to the base folder
cd ..

# Run the MATLAB script for unit_match_merge_ks4_one_probe
matlab -nodisplay -nosplash -r "mouse='$mouse'; date='$save_date'; base_folder='$base_folder'; run('unit_match_merge_ks4_one_probe.m'); exit;"
matlab -nodisplay -nosplash -r "mouse='$mouse'; date='$save_date'; base_folder='$base_folder'; run('unit_match_merge_ks3_one_probe.m'); exit;"
# Run the MATLAB script for the ks3 version (assuming the script name and required adjustments)
# matlab -nodisplay -nosplash -r "mouse='$mouse'; date=''; base_folder=''; run('your_ks3_script_name.m'); exit;"

# Navigate to the folder containing godzilla_merging.py
cd godzilla

# Run the second Python script with inputs
python godzilla_merging.py $mouse $dates $save_date

# Return to the original directory
cd -