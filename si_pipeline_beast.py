# a script to run the spike interface on Diao's data with visualisation


#import the necessary packages


from si_process_one_probe import si_process_one_probe
#grab recordings from the server to local machine (Beast)
base_folder = '/mnt/rds01/ibn-vision/DATA/SUBJECTS/'
mouse = 'M23087'
dates = ['20231207','20231212']
dst_folder = "/home/lab/spikeinterface_sorting/temp_data/"
job_kwargs = dict(n_jobs=20, chunk_duration='1s', progress_bar=True)
for date in dates:
    si_process_one_probe(base_folder, mouse, date,dst_folder,job_kwargs)

