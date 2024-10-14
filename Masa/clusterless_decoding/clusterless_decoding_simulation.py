# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 08:45:26 2024

@author: masahiro.takigawa
"""
import logging
import os

import matplotlib
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Make analysis reproducible
np.random.seed(0)

# Enable logging
logging.basicConfig(level=logging.INFO)

from replay_trajectory_classification.clusterless_simulation import make_simulated_run_data

# Environment 1
(time, position, sampling_frequency,
 multiunits, multiunits_spikes) = make_simulated_run_data()

# Environment 2
np.random.seed(1)
(time2, position2, sampling_frequency2,
 multiunits2, multiunits_spikes2) = make_simulated_run_data()

##
spike_ind, neuron_ind = np.nonzero(multiunits_spikes2)
spike_ind, neuron_ind = np.nonzero(multiunits_spikes)
fig, axes = plt.subplots(7, 1, figsize=(12, 12), constrained_layout=True, sharex=True)
axes[0].plot(time, position, linewidth=3)
axes[0].set_ylabel('Position (cm)')

axes[1].scatter(time[spike_ind], neuron_ind + 1, color='black', s=2)
axes[1].set_yticks((0, multiunits_spikes.shape[1]))
axes[1].set_ylabel('Tetrode Index')

for tetrode_ind in range(multiunits.shape[-1]):
    axes[2 + tetrode_ind].scatter(time, multiunits[:, 0, tetrode_ind], s=1)
    axes[2 + tetrode_ind].set_ylabel(f'Tetrode {tetrode_ind + 1} \n Channel 1 \n Spike Amplitude')

sns.despine()
axes[-1].set_xlabel('Time (s)')
axes[-1].set_xlim((time.min(), time.max()))

##
plt.scatter(multiunits[:, 1, 0], multiunits[:, 2, 0])
plt.ylabel('Spike Amplitude 2')
plt.xlabel('Spike Amplitude 3')
sns.despine(offset=5)

plt.scatter(multiunits2[:, 1, 0], multiunits[:, 2, 0])
plt.ylabel('Spike Amplitude 2')
plt.xlabel('Spike Amplitude 3')
sns.despine(offset=5)

## Fitting
from replay_trajectory_classification import ClusterlessDecoder, Environment, RandomWalk, estimate_movement_var

movement_var = estimate_movement_var(position, sampling_frequency)

environment = Environment(place_bin_size=np.sqrt(movement_var))
transition_type = RandomWalk(movement_var=movement_var)

clusterless_algorithm = 'multiunit_likelihood'
clusterless_algorithm_params = {
    'mark_std': 24.0,
    'position_std': 6,
}


# If your marks are integers, use this algorithm because it is much faster
# clusterless_algorithm = 'multiunit_likelihood_integer_gpu'
 clusterless_algorithm_params = {
     'mark_std': 1.0,
     'position_std': 12.5,
 }
    

decoder = ClusterlessDecoder(
    environment=environment,
    transition_type=transition_type,
    clusterless_algorithm=clusterless_algorithm,
    clusterless_algorithm_params=clusterless_algorithm_params)

decoder2 = ClusterlessDecoder(
    environment=environment,
    transition_type=transition_type,
    clusterless_algorithm=clusterless_algorithm,
    clusterless_algorithm_params=clusterless_algorithm_params)

decoder2.fit(position2[50001:350000], multiunits2[50001:350000])
decoder.fit(position[50001:350000], multiunits[50001:350000])


time_ind = slice(0, 50000)

results1 = decoder.predict(multiunits[time_ind], time=time[time_ind])
results2 = decoder2.predict(multiunits[time_ind], time=time[time_ind])

fig, axes = plt.subplots(3, 1, sharex=True, constrained_layout=True, figsize=(15, 7))

spike_ind, neuron_ind = np.nonzero(multiunits_spikes[time_ind])
axes[0].scatter(time[spike_ind], neuron_ind + 1, c="black", s=0.5, clip_on=False)
axes[0].set_yticks((1, multiunits_spikes.shape[1]))
axes[0].set_ylabel('Tetrodes')
axes[0].set_title("Multiunit Spikes")

results.causal_posterior.plot(x="time", y="position", ax=axes[1], cmap="bone_r", vmin=0.0, vmax=0.05, clip_on=False)
axes[1].plot(time[time_ind], position[time_ind], color="magenta", linestyle="--", linewidth=3, clip_on=False)
axes[1].set_xlabel("")
axes[1].set_title("Causal Posterior Probability of Position")
results.acausal_posterior.plot(x="time", y="position", ax=axes[2], cmap="bone_r", vmin=0.0, vmax=0.05, clip_on=False)
axes[2].plot(time[time_ind], position[time_ind], color="magenta", linestyle="--", linewidth=3, clip_on=False)
axes[2].set_title("Acausal Posterior Probability of Position")
axes[2].set_xlabel('Time [s]')
sns.despine(offset=5)


plt.imshow(results1.acausal_posterior[0:50000].T,extent=[-1, 1, -1, 1])
plt.show()
plt.imshow(results2.acausal_posterior[0:50000].T,extent=[-1, 1, -1, 1])
plt.show()