
# get all the recordings on that day
# allocate destination folder and move the ephys folder on the server to Beast lab user
from pathlib import Path

import os
import shutil


from datetime import datetime
import itertools

startTime = datetime.now()
print('Start Time:' + startTime.strftime("%m/%d/%Y, %H:%M:%S"))
''' this section defines the animal and dates and fetch the recordings from the server to Beast'''
import sys
# The first command-line argument after the script name is the mouse identifier.
#mouse='M24019' #mouse id
#save_date='20240624' #date of recording
#dates='20240624/20240624_0' #acquisition date and session e.g. dates='20240624/20240624_0,20240624/20240624_1'
#base_folder='/home/lab/spikeinterface_sorting/temp_data/'  # Adjust this path if necessary
#local_folder = base_folder
#no_probe=1 #number of probes you have in this session

# The first command-line argument after the script name is the mouse identifier.
mouse = sys.argv[1]
# All command-line arguments after `mouse` and before `save_date` are considered dates.
dates = sys.argv[2].split(',')   # This captures all dates as a list.
# The last command-line argument is `save_date`.
save_date = sys.argv[3]
local_folder = sys.argv[4]
no_probe = sys.argv[5]
print(mouse)
print('acquisition folder: ',dates)
use_ks4 = sys.argv[6].lower() in ['true', '1', 't', 'y', 'yes']
use_ks3 = sys.argv[7].lower() in ['true', '1', 't', 'y', 'yes']
base_folder = '/mnt/rds01/ibn-vision/DATA/SUBJECTS/'

print(mouse)
print(dates)
print(save_date)

save_folder = local_folder+ mouse +"/" + save_date +"/"
# get all the recordings on that day
print(save_folder)

import os
import subprocess
def sorting_key(s):
    return int(s.split('_g')[-1])

#grab recordings from the server to local machine
print(dates)
g_files_all = []
# iterate over all directories in source folder
date_count = 0
for date in dates:
    print('acquisition folder:',date)
    date_count = date_count + 1
    ephys_folder = base_folder + mouse + '/ephys/' + date +'/'
    dst_folder = save_folder
    ephys_folder = base_folder + mouse + '/ephys/' + date +'/'
    g_files = []
    print('copying ephys data from:' + ephys_folder)
    for dirname in os.listdir(ephys_folder):
    #     # check if '_g' is in the directory name
    #     #only grab recording folders - there might be some other existing folders for analysis or sorted data
        if '_g' in dirname:
    #         # construct full directory path
            g_files.append(dirname)
            source = os.path.join(ephys_folder, dirname)
            destination = os.path.join(dst_folder, dirname)
            # copy the directory to the destination folder
            shutil.copytree(source, destination)
    print('Start to copying files to local folder: ')
    print(datetime.now() - startTime)


