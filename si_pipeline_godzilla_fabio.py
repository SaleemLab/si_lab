

#import the necessary packages


from si_process_fabio_2 import si_process_fabio_2
import os
import subprocess
from si_process_fabio_one_probe import si_process_fabio_one_probe

subprocess.run('ulimit -n 4096',shell=True)
#grab recordings from the server to local machine (Beast)
base_folder = '/mnt/rds01/ibn-vision/DATA/SUBJECTS/'
mouse = 'M24029'
dates = ['20240424']


job_kwargs = dict(n_jobs=32, chunk_duration='1s', progress_bar=True)
for date in dates:
    dst_folder = "/home/saleem_lab/spikeinterface_sorting/temp_data/" + date + '/'
    #pathforprobe = base_folder + '/' + mouse + '/' + 'ephys' + '/' +  date + '/' +  mouse+'_'+date+'_g0_imec1'
    os.makedirs(dst_folder, exist_ok=True)
    import os.path
    # if os.path.isdir(pathforprobe):
    si_process_fabio_2(base_folder, mouse, date, dst_folder, job_kwargs)
    #else:
    #si_process_fabio_one_probe(base_folder, mouse, date,dst_folder,job_kwargs)
