#!/bin/bash

#!/bin/bash

# Define variables
mouse='M24086' #mouse id
save_date='20250115' #date of recording
base_folder='/home/lab/spikeinterface_sorting/temp_data/'  # local folder of godzilla
no_probe=1 #number of probes you have in this session
use_ks4=true #use kilosort4 
use_ks3=true #use kilosort3
server_folder='/home/lab/rds01/ibn-vision/DATA/SUBJECTS/' #server folder where the data is stored
g_files_to_ignore='[[],[0,1]]' #files to ignore for each probe
# Run the first Python script with inputs
python premerging_zarr.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder


# Run the MATLAB script for unit_match_merge_ks4_one_probe
matlab -nodisplay -nosplash -r "mouse='${mouse}'; date='${save_date}'; base_folder='${base_folder}';noprobe='${no_probe}'; run('unit_match_merge_ks4.m'); exit;"
matlab -nodisplay -nosplash -r "mouse='${mouse}'; date='${save_date}'; base_folder='${base_folder}';noprobe='${no_probe}'; run('unit_match_merge_ks3.m'); exit;"
# Run the MATLAB script for the ks3 version (assuming the script name and required adjustments)
# matlab -nodisplay -nosplash -r "mouse='$mouse'; date=''; base_folder=''; run('your_ks3_script_name.m'); exit;"



# Run the second Python script with inputs
python merging.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3



