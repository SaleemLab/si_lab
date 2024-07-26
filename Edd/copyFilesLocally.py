# Move ephys directories from server to local drive

import shutil
from datetime import datetime
import sys

startTime = datetime.now()
print('Start Time:' + startTime.strftime("%m/%d/%Y, %H:%M:%S"))

''' this section defines the animal and dates and fetch the recordings from the server to Beast'''
mouse = sys.argv[1] # The first command-line argument after the script name is the mouse identifier.
dates = sys.argv[2].split(',')   # This captures all dates as a list.
save_date = sys.argv[3]  # Defines local save location
local_folder = sys.argv[4] # Defines local save location
no_probe = sys.argv[5]
use_ks4 = sys.argv[6].lower() in ['true', '1', 't', 'y', 'yes']
use_ks3 = sys.argv[7].lower() in ['true', '1', 't', 'y', 'yes']
server_path = '/mnt/rds01/ibn-vision/DATA/SUBJECTS/'

print('Acquisition folders: ', dates)
print('Mouse: ', mouse)

save_folder = local_folder + mouse + "/" + save_date + "/"
# get all the recordings on that day
print('Local directory location: ', save_folder)


#grab recordings from the server to local machine
print(dates)
g_files_all = []
# iterate over all directories in source folder
date_count = 0
for date in dates:
    date_count = date_count + 1
    ephys_folder = server_path + mouse + '/ephys/' + date + '/'
    print('copying ephys data from: ', ephys_folder, ' to: ', save_folder)
    shutil.copytree(ephys_folder, save_folder)

print('Time to copy files to local folder: ')
print(datetime.now() - startTime)


