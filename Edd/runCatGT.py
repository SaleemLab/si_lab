


import os
import shutil
from datetime import datetime

startTime = datetime.now()
print('Start Time:' + startTime.strftime("%m/%d/%Y, %H:%M:%S"))
import sys

# The first command-line argument after the script name is the mouse identifier.
mouse='M24019' #mouse id
save_date='20240716' #date of recording
dates='20240716/20240716_0,20240716/20240716_2' #acquisition date and session e.g. dates='20240624/20240624_0,20240624/20240624_1'
base_folder='/home/lab/spikeinterface_sorting/temp_data/'  # Adjust this path if necessary
local_folder = base_folder
no_probe=1 #number of probes you have in this session

# The first command-line argument after the script name is the mouse identifier.
#ouse = sys.argv[1]
# All command-line arguments after `mouse` and before `save_date` are considered dates.
#dates = sys.argv[2].split(',')   # This captures all dates as a list.
# The last command-line argument is `save_date`.
#save_date = sys.argv[3]
#local_folder = sys.argv[4]
#no_probe = sys.argv[5]
print(mouse)
print('acquisition folder: ',dates)
#use_ks4 = sys.argv[6].lower() in ['true', '1', 't', 'y', 'yes']
#use_ks3 = sys.argv[7].lower() in ['true', '1', 't', 'y', 'yes']
base_folder = '/mnt/rds01/ibn-vision/DATA/SUBJECTS/'
base_folder = 'D:/TestData/'
local_folder = base_folder

print(mouse)
print(dates)
print(save_date)

save_folder = local_folder+ mouse +"/"
print('save folder: ',save_folder)
dates = dates.split(',')
nAcq = (len(dates))
date_count = 0
pathToRunit = 'home/lab/CatGT-linux/runit.sh'
pathToRunit = 'C:/Users/edward.horrocks/Documents/Code/CatGT-win/runit.bat'


osCommands = []

for date in dates:
    print('acquisition folder:',date)
    date_count = date_count + 1
    ephys_folder = save_folder + 'ephys/' + date
    #print('dirtest: ',ephys_folder)

    runName = date.split('/')
    runName = mouse + '_' +runName[1]
    #print('rn: ', runName)




    cmdStr = pathToRunit + ' ' \
             + '-dir=' + ephys_folder + ' -run=' + runName  \
             + ' -g=0,100' + ' -t=0' + ' -t_miss_ok' \
             + ' -prb_fld' + ' -out_prb_fld' + ' -ap' + ' -ni' + ' -prb=0:1' + ' -prb_miss_ok' \
             + ' -dest=' + ephys_folder

    osCommands.append(cmdStr)



print(osCommands[0])
print(osCommands[1])

from subprocess import Popen
import subprocess

procs = [Popen(i) for i in osCommands]
for p in procs:
    p.wait()

if nAcq > 1:  # we also want to run supercat
    print("running supercat ")

    cmdStr = pathToRunit + ' ' \
             + '-supercat='

    for date in dates:
        ephys_folder = save_folder + 'ephys/' + date

        runName = date.split('/')
        baseDate = runName[0]
        runName = mouse + '_' + runName[1]
        cmdStr = cmdStr + '{' + "'" + ephys_folder + '/' + 'catgt_' + runName + '_g0' + "'" + ',' + "'" + runName + "'" + '}'

    cmdStr = cmdStr +  ' -prb_fld -ap -ni -prb=0:1 -prb_miss_ok -supercat_trim_edges -dest=' + save_folder + 'ephys/' + baseDate +'/'

    subprocess.run(cmdStr,check=True)












# get all the recordings on that day

# pass 1 - run CatGT with minimal params on each acquisition (parallel)
# pass 2 - run CatGT supercat on catGT outputs



#.\runit.bat
#-dir=D:\TestData\M24019\ephys\20240716\20240716_0
#-run=M24019_20240716_0
#-g='0,100' -t=0 -t_miss_ok
#-prb_fld -out_prb_fld -ap -ni -prb='0:1' -prb_miss_ok
#-dest=D:\TestData\

