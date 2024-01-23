# si_lab
Developed implementation of spikeinterface on Beast and Godzilla

# Jupyter Notebook - Live Processes with Plotting
On Beast:

`conda activate si_env`

`jupyter notebook --no-browser`

Open Another terminal: 

`ssh -L 8888:localhost:8888 lab@saleem07`


Copy jupyter notebook link to your local PC browser
Open Jupyter notebook: 

`si_with_visualisation_offline_beast.ipy`



# Batch Processing in Background
For batch processing, copy si_pipeline.py and modify the specified animal and date
Before you run it in python, please enter the following command line on Beast:

`ulimit -n 4096`

The waveform extraction process sometimes opens too many files and Linux has a default limit of 1024.

For background processing:

`nohup python si_pipeline.py&`

You can check the messages/errors of the script in the ouput file. Please kill the process when you finish

# What Does This Pipeline Do?

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

# Post-Sorting Analysis Keypoints

## Waveform extraction
The waveform extraction is performed by randomly sampling a subset spikes from the recording for that each unit.

This extracts all waveforms snippets for each unit.

In the ~/waveform/ folder, ~/waveforms/ folder contains 'sample_index_#xx.npy' and 'waveforms_#xx.npy' for each unit.
The sampled_index file contains indexes of the extracted waveforms and the waveforms file contains detailed info of the waveforms (num_spikes, num_sample, num_channel) properties starting from the indexes counting from the sampling number(num_sample).
For example, Unit 10 has 200 spikes extracted from 500 spikes and at index#10 (10th spike) and this waveform starts at index#10 unitl index#10+num_sample.

## Metrics
With extracted waveform:
We can compute all sorts of metrics for each unit based on the waveforms.

Here are explanations of the metrics:

**noise_levels** : median absolute deviation

**spike_amplitudes** : max channel

**principal_components** : a local PCA is fitted for each channel

**template_similarity** : similiarity between units for automerging. Note that cosine similarity does not take into account amplitude differences and is not well suited for high-density probes.

**spike_locations** : location of the spikes using centre of mass method

**unit_locations** : This extension is similar to the spike_locations, but instead of estimating a location for each spike based on individual waveforms, it calculates at the unit level using templates

**correlograms** : This extension computes correlograms (both auto- and cross-) for spike trains. The computed output is a 3d array with shape (num_units, num_units, num_bins) with all correlograms for each pair of units (diagonals are auto-correlograms)

**isi_histograms** : This extension computes the histograms of inter-spike-intervals

**template_metrics** : 
**“peak_to_valley”**: duration between negative and positive peaks
**“halfwidth”**: duration in s at 50% of the amplitude
**“peak_to_trough_ratio”:** ratio between negative and positive peaks
**“recovery_slope”**: speed in V/s to recover from the negative peak to 0
**“repolarization_slope”**: speed in V/s to repolarize from the positive peak to 0
**“num_positive_peaks”**: the number of positive peaks
**“num_negative_peaks”**: the number of negative peaks
![image](https://github.com/SaleemLab/si_lab/assets/61952184/da14060e-7f2d-46de-9de3-ed9be1348dc4)

## Quality Metrics
https://allensdk.readthedocs.io/en/latest/_static/examples/nb/ecephys_quality_metrics.html
https://github.com/AllenInstitute/ecephys_spike_sorting/tree/main/ecephys_spike_sorting/modules/quality_metrics
https://spikeinterface.readthedocs.io/en/latest/modules/qualitymetrics.html
Similar as above

**Firing rate** : the total number of spikes divided by the number of seconds in the recording. Log distribution is helpful

>How it can be biased
>
>If a unit is poorly isolated, the firing rate will be over-estimated, because contaminating spikes will be included in the calculation
>If a unit's amplitude is close to threshold, the firing rate will be under-estimated, because some spikes will be missing
>If a unit drifts out of the recording, the firing rate will be under-estimated, because spikes will not be detected for a portion of the recording
>If data acquisition is interrupted (true for a small subset of experiments), the firing rate will be under-estimated, because spikes will be missing from gaps in the recording
>
>How it should be used
>
>Firing rate can be used to filter out units that have too few spikes to result in meaningful analysis. In this case, it may be better to use the firing rate for the specific interval you're analyzing, because some units may drift out of the recording at other times.
>High firing rate units tend to be easier to isolate, since there are more spikes available for fitting the template in Kilosort2. However, there are other metrics that measure isolation more directly and would likely to be better to use instead.

**Presence ratio** : probably not suitable for Diao and Masa

>How it can be biased
>
>Just because a unit has a high presence ratio doesn't mean it's immune to drift. If a unit's amplitude drifts closer to the spike detection threshold, it can result in dramatic changes in apparent firing rate, even if the underlying physiology remains the same.
>Sometimes a low presence ratio can result from highly selective spiking patterns (e.g., firing only during running epochs)
>
>How it should be used
>
>If you are analyzing changes in firing rate over the entire recording session, or are comparing responses to stimuli presented at the beginning and end of the experiment, presence ratio is a simple way to exclude units that would bias your results. However, you should >also look at other quality metrics, such as amplitude cutoff, to check for more subtle effects of electrode drift.
>If you are only analyzing a short segment of the experiment, it may be helpful to disable the default presence ratio filter, in order to maximize the number of units available to you.
>If you're unsure whether a unit has a low presence ratio due to electrode drift or selective firing, plotting its spike amplitudes over time can be informative.

**Amplitude cutoff** : A histogram of spike amplitudes is created and deviations from the expected symmetrical distribution are identified. Deviations from the expected Gaussian distribution are used to estimate the number of spikes missing from the unit. This yields an estimate of the number of spikes missing from the unit (false negative rate). A smaller value for this metric is preferred, as this indicates fewer false negatives. The distribution can be computed on chunks for larger recording, as drift can impact the spike amplitudes (and thus not give a Gaussian distribution anymore).

>How it can be biased:
>
>The calculation assumes that the amplitude histogram is symmetrical (i.e., it uses the upper tail of the distribution to estimate the fraction of spikes missing from the lower tail). If a unit's waveform amplitude changes as a result of electrode drift, this >assumption is usually invalid. 
>Amplitude cutoff is only weakly correlated with other measures of unit quality, meaning it's possible to have well-isolated units with high amplitude cutoff.
>
>How it should be used:
>
>If you are performing analyses that depends on precise measurements of spike timing, setting a low amplitude cutoff threshold (0.01 or lower) is recommended. This will remove a large fraction of units, but will ensure that the unit of interest contain most of the >relevant spikes.

**Inter-spike-interval (ISI) violations** : Neurons have a refractory period after a spiking event during which they cannot fire again. Inter-spike-interval (ISI) violations refers to the rate of refractory period violations (as described by [Hill]). The calculation works under the assumption that the contaminant events happen randomly or come from another neuron that is not correlated with our unit. A correlation will lead to an overestimation of the contamination, whereas an anti-correlation will lead to an underestimation.
(Lobet implementation (rp_violation) Calculates the number of refractory period violations. This is similar (but slightly different) to the ISI violations. The key difference being that the violations are not only computed on consecutive spikes.)

>How it can be biased:
>
>As with all metrics, ISI violations may not be stable throughout the experiment. It may be helpful to re-calculate it for the specific epochs you're analyzing.
>Two neurons with similar waveforms, but firing in largely non-overlapping epochs, could end up being merged into the same cluster. In this case, the ISI violations may be low, even though the resulting unit is a highly contaminated. This situation would tricky to ?>catch, but fortunately shouldn't happen very often.
>How it should be used:
>
>Setting your ISI violations threshold to 0 (or close to it), will help ensure that contaminated units don't make it into your analysis, but will greatly reduce the number of units available. You should think carefully about what degree of contamination your analysis >can tolerate without biasing your conclusions. For example, if you are comparing firing rates of individual units across areas, you'll want to set a low ISI violations threshold to prevent contaminating spikes from affecting your estimates. On the other hand, if >you're comparing overall firing rates between areas, counting spikes from contaminated clusters may be valid.

**SNR** : Signal-to-noise ratio, or SNR, is another classic metric of unit quality. It measures the ratio of the maximum amplitude of the mean spike waveform to the standard deviation of the background noise on one channel. Even though it's widely used in the literature, we don't recommend using it on Neuropixels data for two reasons:

It only takes into account the unit's peak channel, despite the fact that waveforms are often spread across a dozen channels or more.

If the waveform changes due to drift, peak channel SNR can change dramatically, even though overall isolation quality remains consistent.

**Isolation distance** :
You can imagine each unit's PCs a clusters in a 32 x 3 = 96-dimensional space. Isolation distance calculates the size of the 96-dimensional sphere that includes as many "other" spikes as are contained in the original unit's cluster, after normalizing the clusters by their standard deviation in each dimension (Mahalanobis distance). The higher the isolation distance, the more a unit is separated from its neighbors in PC space, and therefore the lower the likelihood that it's contamined by spikes from multiple units.

>How it can be biased
>
>Isolation distance is not immune to drift; if a unit's waveform changes as a result of electrode motion, it could reduce isolation distance without necessarily causing the unit to become more contaminated.
>The exact value of isolation distance will depend on the number of PCs used in the calculation; therefore, it's difficult to compare this metric to previous reports in the literature.
>
>How it should be used
>
>Isolation distance is correlated with overall cluster quality, but it's not a direct measure of contamination rate. For this reason, it should be used in conjunction with other metrics, such as isi_violations, that more directly measure the likelihood of contaminating >spikes.

**d-prime** : Like isolation distance, d-prime is another metric calculated for the waveform PCs. It uses linear discriminant analysis to calculate the separability of one unit's PC cluster and all of the others. A higher d-prime value indicates that the unit is better isolated from its neighbors.

>How it can be biased:
>Like isolation distance, d-prime is not tolerant to drift. Since a single value of d-prime is computed for the entire session, the d-prime value is actually a lower bound on the true value of this metric computed at any one timepoint.
>
>How it should be used:
>d-prime, in principal, gives you an estimate of the false positive rate for each unit. However, more work is required to validate this.


















## Sorter Comparison
Refer these: https://spikeinterface.readthedocs.io/en/latest/modules_gallery/comparison/plot_5_comparison_sorter_weaknesses.html#sphx-glr-modules-gallery-comparison-plot-5-comparison-sorter-weaknesses-py
https://spikeinterface.readthedocs.io/en/latest/modules/comparison.html

Results:
Matching firing events

Compute agreement score:agreement_score[i, k] = match_event_count[i, k] / (event_counts_GT[i] + event_counts_ST[k] - match_event_count[i, k])

Match units - take matched units only?


