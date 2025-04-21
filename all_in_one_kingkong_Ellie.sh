
#!/bin/bash
mouse='M00014' #mouse id
save_date='20250317' #date of recording
#dates='20241220/20241220_0,20241220/20241220_1' #acquisition date and session e.g. dates='20240624/20240624_0,20240624/20240624_1'
base_folder='/home/masa/spikesorting_temp_data/'  # local folder of kingkong
no_probe=1 #number of probes you have in this session
use_ks4=true #use kilosort4 
use_ks3=true #use kilosort3
server_folder='/bendor/Ellie/DATA/SUBJECTS/'
g_files_to_ignore='[[],[0,1,2,3]]' #files to ignore for each probe

# Run the first Python script with inputs

# comment out download_zarr_kingkong.py if this step completed successfully (copying of zarr files to Bendor24 temp folder)
python download_zarr_kingkong.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder

# as of 14th April, the premerging_zarr_premerging script has been broken down into two codes, preprocessing and then spikesorting
python preprocessing_zarr_kingkong_probe0.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
python spikesorting_zarr_kingkong_probe0.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder

#python premerging_zarr_kingkong_probe0.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder 
#python premerging_zarr_kingkong_probe1.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder 
#python premerging_zarr_kingkong.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder $g_files_to_ignore
#python premerging_zarr.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder $g_files_to_ignore

#python upload_kingkong_probe1.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder
python upload_kingkong_probe0.py $mouse $save_date $base_folder $no_probe $use_ks4 $use_ks3 $server_folder

