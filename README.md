# si_lab
Developed implementation of spikeinterface on Beast and Godzilla

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

## What Does This Pipeline Do?

1. Files are moved to local linux machine for offline processing.
2. Apply several steps of preprocessing:
     Highpass filter at 300Hz
     
     Detect bad channels and remove
     
     Phase shift correction
     
     common median reference
   
4. Concatenate recordings from same session
5. Motion correction - non-rigid method
6. Save binary files of the preprocessed and motion corrected recordings (under folder ~/probe0_preprocessed/)
7. Run multiple sorters on the recordings - currently only Kilosort3 and Kilosort2.5 (under folders ~/probe0/sorters/kilosort3/)
8. Remove duplicate units from sortings
9. Compare sorters - generate agreement score matrix and a plot inside ~/probe0/sorters/
10. Extract waveforms from individual sorting output folder (under folders ~/probe0/waveform/kilosort3/)
11. Compute metrics under folders ~/probe0/waveform/kilosort3/:
      template metrics - peak_to_valley; peak_trough_ratio; halfwidth; repolarization_slope; recovery_slope; num_positive_peaks; num_negative_peaks
      noise levels
      PCA
      template similarity
      correlograms
      spike amplitudes
      unit locations
      spike locations
      isi histograms

12. Compute quality metrics same as 11
13. Auto + manual curation based on metrics and phy (to be decided and developed)
