# This script is used to compress the raw data files into corresponding folders.
# e.g. you have 3 acquisitions - 20240715_0, 20240715_1, 20240715_2 
#then you will have 3 folders with the same number of files in each folder.
# Also original meta fiels, LFP and sync channels are saved in the original folder.
# Once you are happy with the compression + LFP + sync + meta files, please delete the binary files on the server.

# Define variables
mouse='M24016' #mouse id
save_date='[[20240626],[20240701],[20240706]]' #date of recording
acquisitions='[[0],[0],[0]]' #acquisitions you have so there will be same number of compression folders
base_folder='/mnt/rds01/ibn-vision/DATA/SUBJECTS/'  # Adjust this path if necessary
no_probe=2 #number of probes you have in this session


# Run the first Python script with inputs
python compression.py $mouse $acquisitions $save_date $base_folder $no_probe 

# LFP and sync channels extraction
matlab -nodisplay -nosplash -r "mouse='${mouse}'; date='${save_date}'; base_folder='${base_folder}';noprobe='${no_probe}'; run('LFP_sync_extraction.m'); exit;"