# a script to run the spike interface on Diao's data with visualisation


#import the necessary packages


import os
from si_process import si_process
#grab recordings from the server to local machine (Beast)
base_folder = '/mnt/rds01/ibn-vision/DATA/SUBJECTS/'
mouse = 'M23017'
dates = ['20230629']
dst_folder = "/home/lab/spikeinterface_sorting/temp_data/"
job_kwargs = dict(n_jobs=20, chunk_duration='1s', progress_bar=True)
for date in dates:
    dst_folder = "/home/lab/spikeinterface_sorting/temp_data/" + date + '/'
    os.makedirs(dst_folder, exist_ok=True)
    si_process(base_folder, mouse, date, dst_folder, job_kwargs)

