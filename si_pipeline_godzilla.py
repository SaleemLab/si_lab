# a script to run the spike interface on Diao's data with visualisation


# a script to run the spike interface on Diao's data with visualisation


#import the necessary packages


from si_process import si_process
#grab recordings from the server to local machine (Beast)
base_folder = '/mnt/rds01/ibn-vision/DATA/SUBJECTS/'
mouse = 'M23034'
dates = ['20230805','20230806','20230807']
dst_folder = "/home/saleem_lab/spikeinterface_sorting/temp_data/"

for date in dates:
    si_process(base_folder, mouse, date,dst_folder)