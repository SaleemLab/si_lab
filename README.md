# si_lab
Developed implementation of spikeinterface on Beast

## Jupyter Notebook - Live Processes with Plotting
On Beast:

`conda activate si_env`

`jupyter notebook --no-browser`

Open Another terminal: 

`ssh -L 8888:localhost:8888 lab@saleem07`


Copy jupyter notebook link to your local PC browser
Open Jupyter notebook: 

`si_with_visualisation_offline_beast.ipy`



## Batch Processing in Background
For batch processing, copy si_pipeline.py and modify the specified animal and date
Before you run it in python, please enter the following command line on Beast:

`ulimit -n 4096`

The waveform extraction process sometimes opens too many files and Linux has a default limit of 1024.

For background processing:

`nohup python si_pipeline.py&`

You can check the messages/errors of the script in the ouput file. Please kill the process when you finish
