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
>
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
>
>Like isolation distance, d-prime is not tolerant to drift. Since a single value of d-prime is computed for the entire session, the d-prime value is actually a lower bound on the true value of this metric computed at any one timepoint.
>
>How it should be used:
>
>d-prime, in principal, gives you an estimate of the false positive rate for each unit. However, more work is required to validate this.


**Nearest-neighbors hit rate**: Nearest-neighbors hit rate is another PC-based quality metric. It's derived from the 'isolation' metric originally reported in Chung, Magland et al. (2017). This metric looks at the PCs for one unit and calculates the fraction of their nearest neighbors that fall within the same cluster. If a unit is highly contaminated, then many of the closest spikes will come from other units. Nearest-neighbors hit rate is nice because it always falls between 0 and 1, making it straightforward to compare across different datasets.

>How it can be biased
>
>Like the other PC-based metrics, nn_hit_rate can be negatively impacted by electrode drift.
>
>How it should be used
>
>nn_hit_rate is a nice proxy for overall cluster quality, but should be used in conjunction with other metrics that measure missing spikes or contamination rate more directly.


**Silhouette score** : Gives the ratio between the cohesiveness of a cluster and its separation from other clusters. Values for silhouette score range from -1 to 1.

>Expectation and use
>
>A good clustering with well separated and compact clusters will have a silhouette score close to 1. A low silhouette score (close to -1) indicates a poorly isolated cluster (both type I and type II error). SpikeInterface provides access to both implementations of >silhouette score.
>
>To reduce complexity the default implementation in SpikeInterface is to use the simplified silhouette score. This can be changes by switching the silhouette method to either ‘full’ (the Rousseeuw implementation) or (‘simplified’, ‘full’) for both methods when entering >the qm_params parameter.


**Sliding refractory period violations** : Compute maximum allowed refractory period violations for all possible refractory periods in recording. Bins of 0.25ms are used in the [IBL] implementation. For each bin, this maximum value is compared with that observed in the recording. In the [IBL] implementation, a threshold is imposed and a binary value returned (based on whether the unit ‘passes’ the metric). The SpikeInterface implementation, instead, returns the minimum contamination with at least 90% confidence. This contamination value is between 0 and 1.

>Expectation and use
>
>Similar to the ISI violations metric, this metric quantifies the number of refractory period violations seen in the recording of the unit. This is an estimate of the false positive rate. A high number of violations indicates contamination, so a low value is expected >for high quality units.

**amplitude CV (coefficient of variation)** : a measure of the amplitude variability. It is computed as the ratio between the standard deviation and the amplitude mean. To obtain a better estimate of this measure, it is first computed separately for several temporal bins. Out of these values, the median and the range (percentile distance, by default between the 5th and 95th percentiles) are computed.

>Expectation and use
>
>The amplitude CV median is expected to be relatively low for well-isolated units, indicating a “stereotypical” spike shape.
>
>The amplitude CV range can be high in the presence of noise contamination, due to amplitude outliers like in the example below.


**Amplitude median** : Geometric median amplitude is computed in the log domain. The metric is then converted back to original units.

>Expectation and use
>
>A larger value (larger signal) indicates a better unit.



**Firing range** : The firing range indicates the dispersion of the firing rate of a unit across the recording. It is computed by taking the difference between the 95th percentile’s firing rate and the 5th percentile’s firing rate computed over short time bins (e.g. 10 s).

>Expectation and use
>
>Very high levels of firing ranges, outside of a physiological range, might indicate noise contamination.

**L-ratio** : The Mahalanobis distance and chi-squared inverse cdf (given the assumption that the spikes in the cluster distribute normally in each dimension) are used to find the probability of cluster membership for each spike. L-ratio uses 4 principal components (PCs) for each tetrode channel (the first being energy, the square root of the sum of squares of each sample in the waveform, followed by the first 3 PCs of the energy normalised waveform). This yields spikes which are each represented as a point in 16 dimensional space.

>Expectation and use
>
>Since this metric identifies unit separation, a high value indicates a highly contaminated unit (type I error) ([Schmitzer-Torbert] et al.). [Jackson] et al. suggests that this measure is also correlated with type II errors (although more strongly with type I errors).
>
>A well separated unit should have a low L-ratio ([Schmitzer-Torbert] et al.).

**Standard Deviation (SD) ratio** : All spikes from the same neuron should have the same shape. This means that at the peak of the spike, the standard deviation of the voltage should be the same as that of noise. If spikes from multiple neurons are grouped into a single unit, the standard deviation of spike amplitudes would likely be increased.

>Expectation and use
>
>For a unit representing a single neuron, this metric should return a value close to one. However for units that are contaminated, the value can be significantly higher.

**Synchrony Metrics** : This function is providing a metric for the presence of synchronous spiking events across multiple spike trains.

The complexity is used to characterize synchronous events within the same spike train and across different spike trains. This way synchronous events can be found both in multi-unit and single-unit spike trains. Complexity is calculated by counting the number of spikes (i.e. non-empty bins) that occur at the same sample index, within and across spike trains.

Synchrony metrics can be computed for different synchrony sizes (>1), defining the number of simultaneous spikes to count.

>Expectation and use
>
>A larger value indicates a higher synchrony of the respective spike train with the other spike trains. Larger values, especially for larger sizes, indicate a higher probability of noisy spikes in spike trains.

**Drift metrics (drift_ptp, drift_std, drift_mad)** : 
Geometric positions and times of spikes within the cluster are estimated. Over the duration of the recording, the drift observed in positions of spikes is calculated in intervals, with respect to the overall median positions over the entire recording. These are referred to as “drift signals”.

The drift_ptp is the peak-to-peak of the drift signal for each unit.

The drift_std is the standard deviation of the drift signal for each unit.

The drift_mad is the median absolute deviation of the drift signal for each unit.

The SpikeInterface implementation differes from the original Allen because it uses spike location estimates (using compute_spike_locations() - either center of mass or monopolar triangulation), instead of the center of mass of the first PC projection. In addition the Allen Institute implementation assumes linear and equally spaced arrangement of channels.

Finally, the original “cumulative_drift” and “max_drift” metrics have been refactored/modified for the following reasons:

“max_drift” is calculated with the peak-to-peak, so it’s been renamed “drift_ptp”

“cumulative_drift” sums the absolute value of the drift signal for each interval. This makes it very sensitive to the number of bins (and hence the recording duration)! The “drift_std” and “drift_mad”, instead, are measures of the dispersion of the drift signal and are insensitive to the recording duration.

>Expectation and use
>
>Drift metrics represents how much, in um, a unit has moved over the recording. Larger values indicate more “drifty” units, possibly of lower quality.



## Sorter Comparison
Refer these: https://spikeinterface.readthedocs.io/en/latest/modules_gallery/comparison/plot_5_comparison_sorter_weaknesses.html#sphx-glr-modules-gallery-comparison-plot-5-comparison-sorter-weaknesses-py
https://spikeinterface.readthedocs.io/en/latest/modules/comparison.html

Results:
Matching firing events

Compute agreement score:agreement_score[i, k] = match_event_count[i, k] / (event_counts_GT[i] + event_counts_ST[k] - match_event_count[i, k])

Match units - take matched units only?




## Bomb Cell Metrics
https://github.com/Julie-Fabre/bombcell/wiki/Detailed-overview-of-quality-metrics
Consider run the result through bombcell as well? Write a script to merge both metric tables x unit by y metrics and see which units pass all the metrics?

List of all the quality metrics, and associated parameters
-Noise parameters:

Brief description

Number of waveform peaks	

Number of waveform troughs	

Waveform peak-to-trough duration

Waveform spatial decay slope

Waveform baseline flatness

Non-somatic parameters:

Brief description

Main waveform peak precedes trough	

Main waveform peak is larger than trough

Multi-unit parameters:

Brief description	

Number of spikes	

Percentage of spikes below detection threshold

Fraction of refractory period violations

Drift estimate	maxDriftEstimate

Presence ratio	presenceRatio

Mean raw waveform amplitude

Signal-to-noise ratio

Isolation distance

L-ratio

Silhouette score


Quality metric parameters:

Brief description

Number of raw spikes to extract

Save individual raw waveform traces for each unit	

Spike width (in samples) to extract

Extract raw data or not

Time samples in the baseline raw waveforms to use for signal-to-noise ratio

Refractory period time (s)

Refractory period time (s) step

Refractory period time (s)

Refractory period censored time (s)

Divide the recording into time chunks or not	computeTimeChunks

Size of recording time chunks	

Bin size in seconds for presence ratio	

Compute presence ratio or not	computeDrift	

Start time in samples to compute template waveform flatness	

Stop time in samples to compute template waveform flatness	

Threshold to detect template waveform peaks	

Compute distance metrics or not

Number of channels to use for distance metrics	


# Unit Match Implementation

## Waveform Parameters
The amplitude of that weighted-average
waveform (Figure 2E, Equation 10).
• The average centroid (Figure 2F, Equation
6), defined as the average position weighted
by the maximum amplitude on each recording site.
• The trajectory of the spatial centroid from
0.2 ms before the peak to 0.5 ms after the
peak (Figure 2F, Equation 4).
• The distance travelled at each time point
(Figure 2F).
• The travel direction of the spatial centroid at
each time point (Figure 2F, Equation 5). 

## Similarity Scores
• Decay similarity (𝐷; Equation 14); spatial decay.
• Waveform similarity (𝑊, Equation 18);
waveform correlation and normalized difference averaged.
• Amplitude similarity (𝐴; Equation 13)
• Centroid similarity (𝐶, Equation 20)
• Volatility similarity (𝑉, Equation 23); captures the stability of the difference between
centroids over time.
• Route similarity (𝑅; Equation 24): captures
the overall similarity of the trajectory: direction and distance travelled

## Match Probability After Putative Matches and Drift Correction

). Plotting the
distributions of the six similarity scores for these
pairs reveals major differences (Figure 3I). Based
on these distributions, we defined a naïve Bayes
classifier (Equation 29), which takes as input the
values of the six similarity scores for two spike
waveforms and outputs the ‘match probability’:
the posterior probability of the two waveforms
coming from the same unit. 
