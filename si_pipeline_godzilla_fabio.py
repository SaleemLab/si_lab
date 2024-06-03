

#import the necessary packages


from si_process_fabio import si_process_fabio
import os
import subprocess
from si_process_fabio_one_probe import si_process_fabio_one_probe

subprocess.run('ulimit -n 10000',shell=True)
#grab recordings from the server to local machine (Beast)
base_folder = '/mnt/rds01/ibn-vision/DATA/SUBJECTS/'
mouse = 'M23020'
dates = ['20230829', '20230830']

job_kwargs = dict(n_jobs=32, chunk_duration='1s', progress_bar=True)
for date in dates:
    dst_folder = "/home/saleem_lab/spikeinterface_sorting/temp_data/" + date + '/'
    pathforprobe = base_folder + mouse + '/' + 'ephys' + '/' + date + '/' + mouse + '_' + date+'_g0/' + mouse +'_'+ date +'_g0_imec1'

    import os.path
    if os.path.isdir(pathforprobe):
        print('Running dual probe pipeline')
        si_process_fabio(base_folder, mouse, date, dst_folder, job_kwargs)
    else:
        print('Running single probe pipeline')
        si_process_fabio_one_probe(base_folder, mouse, date,dst_folder,job_kwargs)


