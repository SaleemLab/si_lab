
import shutil
import os
import subprocess
from datetime import datetime

#subprocess.run('ulimit -n 4096',shell=True)
def sorting_key(s):
    return int(s.split('_g')[-1])

mouse='M24019' #mouse id
save_date='20240710' #date of recording
dates='20240716/20240716_0,20240716/20240716_2' #acquisition date and session, e.g. dates='20240624/20240624_0,20240624/20240624_1'
dates = dates.split(',')   # This captures all dates as a list.

base_folder='D:/TestData/'  # local folder of godzilla
local_folder = base_folder
print(mouse)
print('acquisition folder: ',dates)
print(save_date)

startTime = datetime.now()
print('Start Time:' + startTime.strftime("%m/%d/%Y, %H:%M:%S"))



print(dates)
g_files_all = []
# iterate over all directories in source folder
date_count = 0
for date in dates:
    print('acquisition folder:',date)
    date_count = date_count + 1
    ephys_folder = base_folder + mouse + '/ephys/' + date +'/'
    dst_folder = local_folder + date + '/'
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
            #shutil.copytree(source, destination)

            g_files = sorted(g_files, key=sorting_key)
            g_files_all = g_files_all + g_files
            print(g_files)
            print('all g files:', g_files_all)



