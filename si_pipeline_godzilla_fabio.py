

#import the necessary packages


import os
import subprocess

subprocess.run('ulimit -n 4096',shell=True)
#grab recordings from the server to local machine (Beast)
base_folder = '/mnt/rds01/ibn-vision/DATA/SUBJECTS/'
mouse = 'M24028'
dates = ['20240427']

job_kwargs = dict(n_jobs=32, chunk_duration='1s', progress_bar=True)
for date in dates:
    dst_folder = "/home/saleem_lab/spikeinterface_sorting/temp_data/" + date + '/'
    os.makedirs(dst_folder, exist_ok=True)
    import os.path
    if os.path.isfile(dst_folder,stream_name = 'imec1.ap'):
        si_process_fabio(base_folder, mouse, date, dst_folder, job_kwargs)
    else:
        si_process_fabio_oneprobe(base_folder, mouse, date,dst_folder,job_kwargs)
