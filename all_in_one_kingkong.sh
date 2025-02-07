#!/bin/bash

# Define variables
mouse='M24017' #mouse id
save_date='20240603' #date of recording
#dates='20241220/20241220_0,20241220/20241220_1' #acquisition date and session e.g. dates='20240624/20240624_0,20240624/20240624_1'
base_folder='/home/masa/spikesorting_temp_data/'  # local folder of godzilla
no_probe=2 #number of probes you have in this session
use_ks4=true #use kilosort4 
use_ks3=true #use kilosort3
server_folder='/home/masa/rds01/ibn-vision/DATA/SUBJECTS/' #server folder where the data is stored
server_folder='/saleem/ibn-vision/DATA/SUBJECTS/'
g_files_to_ignore='[[6,7,8,9]]' #files to ignore for each probe

# Run the first Python script with inputs
python download_zarr_kingkong.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
python premerging_zarr_kingkong_probe0.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
python premerging_zarr_kingkong_probe1.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
#python premerging_zarr_kingkong.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder $g_files_to_ignore
#python premerging_zarr.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder $g_files_to_ignore

# Run the MATLAB script for unit_match_merge_ks4_one_probe
#matlab -nodisplay -nosplash -r "mouse='${mouse}'; date='${save_date}'; base_folder='${base_folder}';noprobe='${no_probe}'; run('unit_match_merge_ks4.m'); exit;"
#matlab -nodisplay -nosplash -r "mouse='${mouse}'; date='${save_date}'; base_folder='${base_folder}';noprobe='${no_probe}'; run('unit_match_merge_ks3.m'); exit;"
# Run the MATLAB script for the ks3 version (assuming the script name and required adjustments)
# matlab -nodisplay -nosplash -r "mouse='$mouse'; date=''; base_folder=''; run('your_ks3_script_name.m'); exit;"

# Run the second Python script with inputs
#python merging_kingkong_probe1.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
#python merging_kingkong_probe0.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
python upload_kingkong_probe1.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
python upload_kingkong_probe0.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder


# Define variables
mouse='M24017' #mouse id
save_date='20240604' #date of recording
#dates='20241220/20241220_0,20241220/20241220_1' #acquisition date and session e.g. dates='20240624/20240624_0,20240624/20240624_1'
base_folder='/home/masa/spikesorting_temp_data/'  # local folder of godzilla
no_probe=2 #number of probes you have in this session
use_ks4=true #use kilosort4 
use_ks3=true #use kilosort3
server_folder='/home/masa/rds01/ibn-vision/DATA/SUBJECTS/' #server folder where the data is stored
server_folder='/saleem/ibn-vision/DATA/SUBJECTS/'
g_files_to_ignore='[[6,7,8,9]]' #files to ignore for each probe

# Run the first Python script with inputs
python download_zarr_kingkong.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
python premerging_zarr_kingkong_probe0.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
python premerging_zarr_kingkong_probe1.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
#python premerging_zarr_kingkong.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder $g_files_to_ignore
#python premerging_zarr.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder $g_files_to_ignore

# Run the MATLAB script for unit_match_merge_ks4_one_probe
#matlab -nodisplay -nosplash -r "mouse='${mouse}'; date='${save_date}'; base_folder='${base_folder}';noprobe='${no_probe}'; run('unit_match_merge_ks4.m'); exit;"
#matlab -nodisplay -nosplash -r "mouse='${mouse}'; date='${save_date}'; base_folder='${base_folder}';noprobe='${no_probe}'; run('unit_match_merge_ks3.m'); exit;"
# Run the MATLAB script for the ks3 version (assuming the script name and required adjustments)
# matlab -nodisplay -nosplash -r "mouse='$mouse'; date=''; base_folder=''; run('your_ks3_script_name.m'); exit;"

# Run the second Python script with inputs
#python merging_kingkong_probe1.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
#python merging_kingkong_probe0.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
python upload_kingkong_probe1.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
python upload_kingkong_probe0.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder



#!/bin/bash
# Define variables
mouse='M24017' #mouse id
save_date='20240606' #date of recording
#dates='20241220/20241220_0,20241220/20241220_1' #acquisition date and session e.g. dates='20240624/20240624_0,20240624/20240624_1'
base_folder='/home/masa/spikesorting_temp_data/'  # local folder of godzilla
no_probe=2 #number of probes you have in this session
use_ks4=true #use kilosort4 
use_ks3=true #use kilosort3
server_folder='/home/masa/rds01/ibn-vision/DATA/SUBJECTS/' #server folder where the data is stored
server_folder='/saleem/ibn-vision/DATA/SUBJECTS/'
g_files_to_ignore='[[6,7,8,9]]' #files to ignore for each probe

# Run the first Python script with inputs
python download_zarr_kingkong.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
python premerging_zarr_kingkong_probe0.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder 
python premerging_zarr_kingkong_probe1.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
#python premerging_zarr_kingkong.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder $g_files_to_ignore
#python premerging_zarr.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder $g_files_to_ignore

python upload_kingkong_probe1.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
python upload_kingkong_probe0.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder


#!/bin/bash
# Define variables
mouse='M24017' #mouse id
save_date='20240601' #date of recording
#dates='20241220/20241220_0,20241220/20241220_1' #acquisition date and session e.g. dates='20240624/20240624_0,20240624/20240624_1'
base_folder='/home/masa/spikesorting_temp_data/'  # local folder of godzilla
no_probe=2 #number of probes you have in this session
use_ks4=true #use kilosort4 
use_ks3=true #use kilosort3
server_folder='/home/masa/rds01/ibn-vision/DATA/SUBJECTS/' #server folder where the data is stored
server_folder='/saleem/ibn-vision/DATA/SUBJECTS/'
g_files_to_ignore='[[6,7,8,9]]' #files to ignore for each probe

# Run the first Python script with inputs
python download_zarr_kingkong.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
python premerging_zarr_kingkong_probe0.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder 
python premerging_zarr_kingkong_probe1.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
#python premerging_zarr_kingkong.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder $g_files_to_ignore
#python premerging_zarr.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder $g_files_to_ignore

python upload_kingkong_probe1.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
python upload_kingkong_probe0.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder


# Define variables
mouse='M24018' #mouse id
save_date='20240708' #date of recording
#dates='20241220/20241220_0,20241220/20241220_1' #acquisition date and session e.g. dates='20240624/20240624_0,20240624/20240624_1'
base_folder='/home/masa/spikesorting_temp_data/'  # local folder of godzilla
no_probe=2 #number of probes you have in this session
use_ks4=true #use kilosort4 
use_ks3=true #use kilosort3
server_folder='/home/masa/rds01/ibn-vision/DATA/SUBJECTS/' #server folder where the data is stored
server_folder='/saleem/ibn-vision/DATA/SUBJECTS/'
g_files_to_ignore='[[6,7,8,9]]' #files to ignore for each probe

# Run the first Python script with inputs
python download_zarr_kingkong.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
python premerging_zarr_kingkong_probe0.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder 
python premerging_zarr_kingkong_probe1.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
#python premerging_zarr_kingkong.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder $g_files_to_ignore
#python premerging_zarr.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder $g_files_to_ignore

python upload_kingkong_probe1.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
python upload_kingkong_probe0.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder




# Define variables
mouse='M24018' #mouse id
save_date='20240711' #date of recording
#dates='20241220/20241220_0,20241220/20241220_1' #acquisition date and session e.g. dates='20240624/20240624_0,20240624/20240624_1'
base_folder='/home/masa/spikesorting_temp_data/'  # local folder of godzilla
no_probe=2 #number of probes you have in this session
use_ks4=true #use kilosort4 
use_ks3=true #use kilosort3
server_folder='/home/masa/rds01/ibn-vision/DATA/SUBJECTS/' #server folder where the data is stored
server_folder='/saleem/ibn-vision/DATA/SUBJECTS/'
g_files_to_ignore='[[6,7,8,9]]' #files to ignore for each probe

# Run the first Python script with inputs
python download_zarr_kingkong.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
python premerging_zarr_kingkong_probe0.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder 
python premerging_zarr_kingkong_probe1.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
#python premerging_zarr_kingkong.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder $g_files_to_ignore
#python premerging_zarr.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder $g_files_to_ignore

python upload_kingkong_probe1.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
python upload_kingkong_probe0.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder

# Define variables
mouse='M24018' #mouse id
save_date='20240715' #date of recording
#dates='20241220/20241220_0,20241220/20241220_1' #acquisition date and session e.g. dates='20240624/20240624_0,20240624/20240624_1'
base_folder='/home/masa/spikesorting_temp_data/'  # local folder of godzilla
no_probe=2 #number of probes you have in this session
use_ks4=true #use kilosort4 
use_ks3=true #use kilosort3
server_folder='/home/masa/rds01/ibn-vision/DATA/SUBJECTS/' #server folder where the data is stored
server_folder='/saleem/ibn-vision/DATA/SUBJECTS/'
g_files_to_ignore='[[6,7,8,9]]' #files to ignore for each probe

# Run the first Python script with inputs
python download_zarr_kingkong.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
python premerging_zarr_kingkong_probe0.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder 
python premerging_zarr_kingkong_probe1.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
#python premerging_zarr_kingkong.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder $g_files_to_ignore
#python premerging_zarr.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder $g_files_to_ignore

python upload_kingkong_probe1.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
python upload_kingkong_probe0.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder

#!/bin/bash
# Define variables
mouse='M24018' #mouse id
save_date='20240712' #date of recording
#dates='20241220/20241220_0,20241220/20241220_1' #acquisition date and session e.g. dates='20240624/20240624_0,20240624/20240624_1'
base_folder='/home/masa/spikesorting_temp_data/'  # local folder of godzilla
no_probe=2 #number of probes you have in this session
use_ks4=true #use kilosort4 
use_ks3=true #use kilosort3
server_folder='/home/masa/rds01/ibn-vision/DATA/SUBJECTS/' #server folder where the data is stored
server_folder='/saleem/ibn-vision/DATA/SUBJECTS/'
g_files_to_ignore='[[],[0,1,2,3]]' #files to ignore for each probe

# Run the first Python script with inputs
python download_zarr_kingkong.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
python premerging_zarr_kingkong_probe0.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder $g_files_to_ignore
python premerging_zarr_kingkong_probe1.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder $g_files_to_ignore
#python premerging_zarr_kingkong.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder $g_files_to_ignore
#python premerging_zarr.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder $g_files_to_ignore

python upload_kingkong_probe1.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
python upload_kingkong_probe0.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder



# Define variables
mouse='M24064' #mouse id
save_date='20241211' #date of recording
#dates='20241220/20241220_0,20241220/20241220_1' #acquisition date and session e.g. dates='20240624/20240624_0,20240624/20240624_1'
base_folder='/home/masa/spikesorting_temp_data/'  # local folder of godzilla
no_probe=2 #number of probes you have in this session
use_ks4=true #use kilosort4 
use_ks3=true #use kilosort3
server_folder='/home/masa/rds01/ibn-vision/DATA/SUBJECTS/' #server folder where the data is stored
server_folder='/saleem/ibn-vision/DATA/SUBJECTS/'
g_files_to_ignore='[[],[0,1,2,3]]' #files to ignore for each probe

# Run the first Python script with inputs
python download_zarr_kingkong.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
python premerging_zarr_kingkong_probe0.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
python premerging_zarr_kingkong_probe1.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
#python premerging_zarr_kingkong.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder $g_files_to_ignore
#python premerging_zarr.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder $g_files_to_ignore

python upload_kingkong_probe1.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
python upload_kingkong_probe0.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder


# Define variables
mouse='M24062' #mouse id
save_date='20241122' #date of recording
#dates='20241220/20241220_0,20241220/20241220_1' #acquisition date and session e.g. dates='20240624/20240624_0,20240624/20240624_1'
base_folder='/home/masa/spikesorting_temp_data/'  # local folder of godzilla
no_probe=2 #number of probes you have in this session
use_ks4=true #use kilosort4 
use_ks3=true #use kilosort3
server_folder='/home/masa/rds01/ibn-vision/DATA/SUBJECTS/' #server folder where the data is stored
server_folder='/saleem/ibn-vision/DATA/SUBJECTS/'
g_files_to_ignore='[[],[0,1,2,3]]' #files to ignore for each probe

# Run the first Python script with inputs
python download_zarr_kingkong.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
python premerging_zarr_kingkong_probe0.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder 
python premerging_zarr_kingkong_probe1.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder 
#python premerging_zarr_kingkong.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder $g_files_to_ignore
#python premerging_zarr.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder $g_files_to_ignore

python upload_kingkong_probe1.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
python upload_kingkong_probe0.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder

# Define variables
mouse='M24062' #mouse id
save_date='20241119' #date of recording
#dates='20241220/20241220_0,20241220/20241220_1' #acquisition date and session e.g. dates='20240624/20240624_0,20240624/20240624_1'
base_folder='/home/masa/spikesorting_temp_data/'  # local folder of godzilla
no_probe=2 #number of probes you have in this session
use_ks4=true #use kilosort4 
use_ks3=true #use kilosort3
server_folder='/home/masa/rds01/ibn-vision/DATA/SUBJECTS/' #server folder where the data is stored
server_folder='/saleem/ibn-vision/DATA/SUBJECTS/'
g_files_to_ignore='[[],[0,1,2,3]]' #files to ignore for each probe

# Run the first Python script with inputs
python download_zarr_kingkong.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
python premerging_zarr_kingkong_probe0.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
python premerging_zarr_kingkong_probe1.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder 
#python premerging_zarr_kingkong.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder $g_files_to_ignore
#python premerging_zarr.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder $g_files_to_ignore

python upload_kingkong_probe1.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
python upload_kingkong_probe0.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder

# Define variables
mouse='M24065' #mouse id
save_date='20250130' #date of recording
#dates='20241220/20241220_0,20241220/20241220_1' #acquisition date and session e.g. dates='20240624/20240624_0,20240624/20240624_1'
base_folder='/home/masa/spikesorting_temp_data/'  # local folder of godzilla
no_probe=2 #number of probes you have in this session
use_ks4=true #use kilosort4 
use_ks3=true #use kilosort3
server_folder='/home/masa/rds01/ibn-vision/DATA/SUBJECTS/' #server folder where the data is stored
server_folder='/saleem/ibn-vision/DATA/SUBJECTS/'
g_files_to_ignore='[[],[0,1,2,3]]' #files to ignore for each probe

# Run the first Python script with inputs
python download_zarr_kingkong.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
python premerging_zarr_kingkong_probe0.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder 
python premerging_zarr_kingkong_probe1.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder 
#python premerging_zarr_kingkong.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder $g_files_to_ignore
#python premerging_zarr.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder $g_files_to_ignore

python upload_kingkong_probe1.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
python upload_kingkong_probe0.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder

# Define variables
mouse='M24065' #mouse id
save_date='20250131' #date of recording
#dates='20241220/20241220_0,20241220/20241220_1' #acquisition date and session e.g. dates='20240624/20240624_0,20240624/20240624_1'
base_folder='/home/masa/spikesorting_temp_data/'  # local folder of godzilla
no_probe=2 #number of probes you have in this session
use_ks4=true #use kilosort4 
use_ks3=true #use kilosort3
server_folder='/home/masa/rds01/ibn-vision/DATA/SUBJECTS/' #server folder where the data is stored
server_folder='/saleem/ibn-vision/DATA/SUBJECTS/'
g_files_to_ignore='[[],[0,1,2,3]]' #files to ignore for each probe

# Run the first Python script with inputs
python download_zarr_kingkong.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
python premerging_zarr_kingkong_probe0.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
python premerging_zarr_kingkong_probe1.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder 
#python premerging_zarr_kingkong.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder $g_files_to_ignore
#python premerging_zarr.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder $g_files_to_ignore

python upload_kingkong_probe1.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
python upload_kingkong_probe0.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder

# Define variables
mouse='M24065' #mouse id
save_date='20250201' #date of recording
#dates='20241220/20241220_0,20241220/20241220_1' #acquisition date and session e.g. dates='20240624/20240624_0,20240624/20240624_1'
base_folder='/home/masa/spikesorting_temp_data/'  # local folder of godzilla
no_probe=2 #number of probes you have in this session
use_ks4=true #use kilosort4 
use_ks3=true #use kilosort3
server_folder='/home/masa/rds01/ibn-vision/DATA/SUBJECTS/' #server folder where the data is stored
server_folder='/saleem/ibn-vision/DATA/SUBJECTS/'
g_files_to_ignore='[[],[0,1,2,3]]' #files to ignore for each probe

# Run the first Python script with inputs
python download_zarr_kingkong.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
python premerging_zarr_kingkong_probe0.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder 
python premerging_zarr_kingkong_probe1.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder 
#python premerging_zarr_kingkong.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder $g_files_to_ignore
#python premerging_zarr.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder $g_files_to_ignore

python upload_kingkong_probe1.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
python upload_kingkong_probe0.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder