#we use UnitMatch to merge units that are similar and recompute the waveforms
#This script loads um_merge_suggestions.mat and merge units suggeted by UnitMatch
# Variable match_id: first column original id, and second column the id to merge with

import os
import numpy as np
import scipy.io as sio
from pathlib import Path
import platform
import os
import shutil
import argparse
import numpy as np
import matplotlib.pyplot as plt
import spikeinterface.sorters
import spikeinterface.full as si
import  scipy.signal
import spikeinterface.extractors as se
import spikeinterface.comparison
from spikeinterface.exporters import export_report
from spikeinterface.exporters import export_to_phy
import spikeinterface.curation
import spikeinterface.widgets as sw
import docker
from datetime import datetime
#load mat file with merge suggestions
job_kwargs = dict(n_jobs=32, chunk_duration='1s', progress_bar=True)
base_folder = '/mnt/rds01/ibn-vision/DATA/SUBJECTS/'
# mice = ['M23032','M23034', 'M23037', 'M23038']
# all_dates = [['20230718','20230719','20230720','20230721','20230722'],
#               ['20230804','20230805','2023806'], 
#               ['2023810','2023811','2023812','2023813'],
#               ['2023816','2023817']]
mice = ['M23034', 'M23037', 'M23038']
all_dates = [['20230806'], 
              ['20230810','20230811','20230812','20230813'],
              ['20230816','20230817']]
mouse_number = 0
for mouse in mice:
    dates = all_dates[mouse_number]
    mouse_number += 1
    for date in dates:
        ephys_folder = base_folder + mouse + '/ephys/' + date +'/'
        analysis_folder = base_folder + mouse + '/analysis/' + date +'/'
        save_folder = base_folder + mouse + '/ephys/' + date +'/'
                
        
        probe0_we_ks3 = si.load_sorting_analyzer(folder = ephys_folder + 'probe0/waveform/kilosort3_merged/') 
        export_report(sorting_analyzer = probe0_we_ks3, output_folder = ephys_folder + 'probe0/waveform/kilosort3_merged_report/',**job_kwargs)

        
        probe1_we_ks3 = si.load_sorting_analyzer(folder = ephys_folder + 'probe1/waveform/kilosort3_merged/') 
        export_report(sorting_analyzer = probe1_we_ks3, output_folder = ephys_folder + 'probe1/waveform/kilosort3_merged_report/',**job_kwargs)



mice = ['M23031']
all_dates = [['20230711','20230712','20230713','20230714']]
mouse_number = 0
for mouse in mice:
    dates = all_dates[mouse_number]
    mouse_number += 1
    for date in dates:
        ephys_folder = base_folder + mouse + '/ephys/' + date +'/'
        analysis_folder = base_folder + mouse + '/analysis/' + date +'/'
        save_folder = base_folder + mouse + '/ephys/' + date +'/'
                
        
        probe0_we_ks3 = si.load_sorting_analyzer(folder = ephys_folder + 'probe0/waveform/kilosort3_merged/') 
        export_report(sorting_analyzer = probe0_we_ks3, output_folder = ephys_folder + 'probe0/waveform/kilosort3_merged_report/',**job_kwargs)

        
        probe1_we_ks3 = si.load_sorting_analyzer(folder = ephys_folder + 'probe1/waveform/kilosort3_merged/') 
        export_report(sorting_analyzer = probe1_we_ks3, output_folder = ephys_folder + 'probe1/waveform/kilosort3_merged_report/',**job_kwargs)
        
             
        

#spikeinterface GUI to check the data
import os
import numpy as np
import scipy.io as sio
from pathlib import Path
import platform
import os
import shutil
import argparse
import numpy as np
import matplotlib.pyplot as plt
import spikeinterface.sorters
import spikeinterface.full as si
import  scipy.signal
import spikeinterface.extractors as se
import spikeinterface.comparison
from spikeinterface.exporters import export_report
from spikeinterface.exporters import export_to_phy
import spikeinterface.curation
import spikeinterface.widgets as sw
import docker
import spikeinterface_gui

from datetime import datetime
base_folder = '/mnt/rds01/ibn-vision/DATA/SUBJECTS/'
mouse = 'M23034'

date = '20230807'
ephys_folder = base_folder + mouse + '/ephys/' + date +'/'
analysis_folder = base_folder + mouse + '/analysis/' + date +'/'
save_folder = base_folder + mouse + '/ephys/' + date +'/'
        

probe0_sorting_ks3 = si.read_sorter_folder(ephys_folder + 'probe0/sorters/kilosort3/')
probe0_we_ks3 = si.load_sorting_analyzer(folder = ephys_folder + 'probe0/waveform/kilosort3_merged/') 
probe0_preprocessed_corrected = si.load_extractor(ephys_folder + 'probe0_preprocessed/')
sw.plot_sorting_summary(sorting_analyzer = probe0_we_ks3, backend = 'spikeinterface_gui')

probe1_sorting_ks3 = si.read_sorter_folder(ephys_folder + 'probe1/sorters/kilosort3/')
probe1_we_ks3 = si.load_sorting_analyzer(folder = ephys_folder + 'probe1/waveform/kilosort3_merged/') 
probe1_preprocessed_corrected = si.load_extractor(ephys_folder + 'probe1_preprocessed/')
export_report(sorting_analyzer = probe1_we_ks3, output_folder = ephys_folder + 'probe1/waveform/kilosort3_merged_report/',**job_kwargs)
    
    