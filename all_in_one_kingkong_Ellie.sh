#!/bin/bash

mouse='M00013' #mouse id
save_date='20250203' #date of recording
#dates='20241220/20241220_0,20241220/20241220_1' #acquisition date and session e.g. dates='20240624/20240624_0,20240624/20240624_1'
base_folder='/home/masa/spikesorting_temp_data/'  # local folder of godzilla
no_probe=1 #number of probes you have in this session
use_ks4=true #use kilosort4 
use_ks3=true #use kilosort3
server_folder='/home/masa/rds01/ibn-vision/DATA/SUBJECTS/' #server folder where the data is stored
server_folder='/bendor/Ellie/DATA/SUBJECTS/'
g_files_to_ignore='[[],[0,1,2,3]]' #files to ignore for each probe

# Run the first Python script with inputs
python download_zarr_kingkong.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
python premerging_zarr_kingkong_probe0.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder 
#python premerging_zarr_kingkong_probe1.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder 
#python premerging_zarr_kingkong.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder $g_files_to_ignore
#python premerging_zarr.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder $g_files_to_ignore

#python upload_kingkong_probe1.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
python upload_kingkong_probe0.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder

