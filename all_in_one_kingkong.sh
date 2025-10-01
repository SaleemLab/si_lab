#!/bin/bash

# Define variables
#mouse='M24018' #mouse id
#save_date='20240717' #date of recording
#dates='20241220/20241220_0,20241220/20241220_1' #acquisition date and session e.g. dates='20240624/20240624_0,20240624/20240624_1'
#base_folder='/home/masa/spikesorting_temp_data/'  # local folder of godzilla
#no_probe=2 #number of probes you have in this session
#use_ks4=true #use kilosort4 
#use_ks3=true #use kilosort3
#server_folder='/run/user/1004/gvfs/smb-share:server=rdp.arc.ucl.ac.uk,share=ritd-ag-project-rd01ie-asale69/ibn-vision/DATA/SUBJECTS/'

#g_files_to_ignore='[[],[0,1,2,3]]' #files to ignore for each probe

#### 1. comment out download_zarr_kingkong.py if this step completed successfully (copying of zarr files to Bendor24 temp folder)
#python download_zarr_kingkong.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder

#### 2 and 3. as of 14th April, the premerging_zarr_premerging script has been broken down into two codes, preprocessing and then spikesorting
#python preprocessing_zarr_kingkong_probe1_temp.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
#python spikesorting_zarr_kingkong_probe1.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
#python preprocessing_zarr_kingkong_probe0_temp.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
#python spikesorting_zarr_kingkong_probe0.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder

#python upload_kingkong_probe1.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
#python upload_kingkong_probe0.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder



# Define variables
mouse='M24064' #mouse id
save_date='20241216' #date of recording
#dates='20241220/20241220_0,20241220/20241220_1' #acquisition date and session e.g. dates='20240624/20240624_0,20240624/20240624_1'
base_folder='/home/masa/spikesorting_temp_data/'  # local folder of godzilla
no_probe=2 #number of probes you have in this session
use_ks4=true #use kilosort4 
use_ks3=true #use kilosort3
server_folder='/run/user/1004/gvfs/smb-share:server=rdp.arc.ucl.ac.uk,share=ritd-ag-project-rd01ie-asale69/ibn-vision/DATA/SUBJECTS/'

g_files_to_ignore='[[],[0,1,2,3]]' #files to ignore for each probe

#### 1. comment out download_zarr_kingkong.py if this step completed successfully (copying of zarr files to Bendor24 temp folder)
#python download_zarr_kingkong.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder

#### 2 and 3. as of 14th April, the premerging_zarr_premerging script has been broken down into two codes, preprocessing and then spikesorting
python preprocessing_zarr_kingkong_probe1.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
python spikesorting_zarr_kingkong_probe1.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
python preprocessing_zarr_kingkong_probe0.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
python spikesorting_zarr_kingkong_probe0.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder

#python upload_kingkong_probe1.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
python upload_kingkong_probe0.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder


# Define variables
mouse='M24064' #mouse id
save_date='20241217' #date of recording
#dates='20241220/20241220_0,20241220/20241220_1' #acquisition date and session e.g. dates='20240624/20240624_0,20240624/20240624_1'
base_folder='/home/masa/spikesorting_temp_data/'  # local folder of godzilla
no_probe=2 #number of probes you have in this session
use_ks4=true #use kilosort4 
use_ks3=true #use kilosort3
server_folder='/run/user/1004/gvfs/smb-share:server=rdp.arc.ucl.ac.uk,share=ritd-ag-project-rd01ie-asale69/ibn-vision/DATA/SUBJECTS/'

g_files_to_ignore='[[],[0,1,2,3]]' #files to ignore for each probe

#### 1. comment out download_zarr_kingkong.py if this step completed successfully (copying of zarr files to Bendor24 temp folder)
#python download_zarr_kingkong.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder

#### 2 and 3. as of 14th April, the premerging_zarr_premerging script has been broken down into two codes, preprocessing and then spikesorting
python preprocessing_zarr_kingkong_probe1.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
python spikesorting_zarr_kingkong_probe1.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
python preprocessing_zarr_kingkong_probe0.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
python spikesorting_zarr_kingkong_probe0.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder

#python upload_kingkong_probe1.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
python upload_kingkong_probe0.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder



# Define variables
mouse='M24064' #mouse id
save_date='20241219' #date of recording
#dates='20241220/20241220_0,20241220/20241220_1' #acquisition date and session e.g. dates='20240624/20240624_0,20240624/20240624_1'
base_folder='/home/masa/spikesorting_temp_data/'  # local folder of godzilla
no_probe=2 #number of probes you have in this session
use_ks4=true #use kilosort4 
use_ks3=true #use kilosort3
server_folder='/run/user/1004/gvfs/smb-share:server=rdp.arc.ucl.ac.uk,share=ritd-ag-project-rd01ie-asale69/ibn-vision/DATA/SUBJECTS/'

g_files_to_ignore='[[],[0,1,2,3]]' #files to ignore for each probe

#### 1. comment out download_zarr_kingkong.py if this step completed successfully (copying of zarr files to Bendor24 temp folder)
#python download_zarr_kingkong.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder

#### 2 and 3. as of 14th April, the premerging_zarr_premerging script has been broken down into two codes, preprocessing and then spikesorting
python preprocessing_zarr_kingkong_probe1.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
python spikesorting_zarr_kingkong_probe1.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
python preprocessing_zarr_kingkong_probe0.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
python spikesorting_zarr_kingkong_probe0.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder

#python upload_kingkong_probe1.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
python upload_kingkong_probe0.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder



# Define variables
mouse='M24064' #mouse id
save_date='20241220' #date of recording
#dates='20241220/20241220_0,20241220/20241220_1' #acquisition date and session e.g. dates='20240624/20240624_0,20240624/20240624_1'
base_folder='/home/masa/spikesorting_temp_data/'  # local folder of godzilla
no_probe=2 #number of probes you have in this session
use_ks4=true #use kilosort4 
use_ks3=true #use kilosort3
server_folder='/run/user/1004/gvfs/smb-share:server=rdp.arc.ucl.ac.uk,share=ritd-ag-project-rd01ie-asale69/ibn-vision/DATA/SUBJECTS/'

g_files_to_ignore='[[],[0,1,2,3]]' #files to ignore for each probe

#### 1. comment out download_zarr_kingkong.py if this step completed successfully (copying of zarr files to Bendor24 temp folder)
#python download_zarr_kingkong.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder

#### 2 and 3. as of 14th April, the premerging_zarr_premerging script has been broken down into two codes, preprocessing and then spikesorting
python preprocessing_zarr_kingkong_probe1.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
python spikesorting_zarr_kingkong_probe1.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
python preprocessing_zarr_kingkong_probe0.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
python spikesorting_zarr_kingkong_probe0.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder

#python upload_kingkong_probe1.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
python upload_kingkong_probe0.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder