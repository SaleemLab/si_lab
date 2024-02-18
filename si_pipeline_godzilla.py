# a script to run the spike interface on Diao's data with visualisation


# a script to run the spike interface on Diao's data with visualisation


#import the necessary packages


from si_process import si_process
from si_process_one_probe import si_process_one_probe
import os
import subprocess
subprocess.run('ulimit -n 4096',shell=True)
#grab recordings from the server to local machine (Beast)
base_folder = '/mnt/rds01/ibn-vision/DATA/SUBJECTS/'
mouse = 'M23031'
dates = ['20230711','20230712','20230713','20230714']

job_kwargs = dict(n_jobs=32, chunk_duration='1s', progress_bar=True)
for date in dates:
    dst_folder = "/home/saleem_lab/spikeinterface_sorting/temp_data/" + date + '/'
    os.makedirs(dst_folder, exist_ok=True)
    si_process_one_probe(base_folder, mouse, date, dst_folder, job_kwargs)
