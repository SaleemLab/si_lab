# a script to run the spike interface on Diao's data with visualisation


#import the necessary packages


from si_process import si_process
#grab recordings from the server to local machine (Beast)
base_folder = '/mnt/rds01/ibn-vision/DATA/SUBJECTS/'
mouse = 'M23038'
dates = ['20230816','20230817']
dst_folder = "/home/lab/spikeinterface_sorting/temp_data/"

for date in dates:
    si_process(base_folder, mouse, date,dst_folder)