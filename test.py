import os
import pandas as pd
mouse = ['M23034']
date = ['20230804','20230805']
base_folder = 'Z:\\ibn-vision\\DATA\\SUBJECTS\\'
for d in date:
    ephys_folder = os.path.join(base_folder,mouse[0],'ephys',d)
    print(ephys_folder)
    g_files = []
    # iterate over all directories in source folder
    for dirname in os.listdir(ephys_folder):
        # check if '_g' is in the directory name
        #only grab recording folders - there might be some other existing folders for analysis or sorted data
        if '_g' in dirname:
            # construct full directory path
            g_files.append(dirname)
            
    print(g_files)
    segment_info_paths = [os.path.join(ephys_folder,'probe0','sorters'), os.path.join(ephys_folder,'probe1','sorters')]
    for segment_info in segment_info_paths:    
        # Load the CSV file
        df = pd.read_csv(segment_info + '\\segment_frames.csv')

        # Add a new column at the front
        df.insert(0, 'segment_info', g_files)
        print(df)
        df.to_csv(segment_info + '\\segment_frames.csv', index=False)


        