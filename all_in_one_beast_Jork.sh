#!/bin/bash

#!/bin/bash

#!/bin/bash

#!/bin/bash

# Define variables
mouse='M24086' #mouse id
save_date='20250122' #date of recording
base_folder='/home/lab/spikeinterface_sorting/temp_data/M24086'  # local folder of godzilla
no_probe=1 #number of probes you have in this session
use_ks4=true #use kilosort4 
#use_ks3=true #use kilosort3
server_folder='/mnt/rd01/ibn-vision/DATA/SUBJECTS/' #server folder where the data is stored
#g_files_to_ignore='[]' #files to ignore for each probe
#g_files_to_ignore='[[0,1],[]]' #files to ignore for each probe
# Run the first Python script with inputs
python premerging_zarr.py $mouse $save_date $base_folder $no_probe $use_ks4 $server_folder 


# Define variables
mouse='M24086' #mouse id
save_date='20250121' #date of recording
base_folder='/home/lab/spikeinterface_sorting/temp_data/M24086'  # local folder of godzilla
no_probe=1 #number of probes you have in this session
use_ks4=true #use kilosort4 
#use_ks3=true #use kilosort3
server_folder='/mnt/rd01/ibn-vision/DATA/SUBJECTS/' #server folder where the data is stored
#g_files_to_ignore='[]' #files to ignore for each probe
#g_files_to_ignore='[[0,1],[]]' #files to ignore for each probe
# Run the first Python script with inputs
python premerging_zarr.py $mouse $save_date $base_folder $no_probe $use_ks4 $server_folder 
