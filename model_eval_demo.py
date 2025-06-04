import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import os
from c_spikes.cascade2p import checks, cascade
from c_spikes.ens2 import ens
from c_spikes.model_eval.model_eval import smooth_spike_train, smooth_prediction, compute_smoothed_correlation

# prediction directories
ens2_dir = "results/ens2_output/predictions"
cascade_dir = "results/cascade_output/predictions"
old_cas = "data/Cascaded_jGCaMP8f_ANM471993_cell03.mat"

# Only one file per directory for now, but this will scale easily
ens2_files = [os.path.join(ens2_dir, f) for f in os.listdir(ens2_dir) if f.endswith('.mat')]
cascade_files = [os.path.join(cascade_dir, f) for f in os.listdir(cascade_dir) if f.endswith('.mat')]
# Load
ens2_data = sio.loadmat(ens2_files[0])
cascade_data = sio.loadmat(cascade_files[0])
old_cas_data = sio.loadmat(old_cas)


# Smooth spike_times and traces
spike_times = ens2_data['spike_times'].flatten()
time_stamps_ens2 = ens2_data['time_stamps'].flatten()
# Subset only those spikes that are in the single ens2 epoch for now
spike_times = spike_times[(spike_times >= time_stamps_ens2[0]) & (spike_times <= time_stamps_ens2[-1])]
spike_times = spike_times - time_stamps_ens2[0]  # start at 0
dff = ens2_data['dff'].flatten()
spike_prob_ens2 = ens2_data['spike_prob'].squeeze()
spike_prob_cascade = cascade_data['spike_prob'].squeeze()
time_stamps_cascade = cascade_data['time_stamps'].squeeze()
spike_prob_ol_cas = old_cas_data['spike_prob'].squeeze()

# Use the same filter width for smoothing as in your config (e.g., 0.05s)
filter_width = 20  # ms
# Sampling, length
sampling_rate_ens2 = 1/(np.mean(np.diff(time_stamps_ens2)))  # Hz
T_ens2 = (len(spike_prob_ens2)-1) / sampling_rate_ens2  # duration in seconds
sampling_rate_cascade = 1/(np.mean(np.diff(time_stamps_cascade[0])))  # Hz
T_cascade = (len(spike_prob_cascade[0])-1) / sampling_rate_cascade  # duration in seconds
# Smooth spike train and prediction traces
smoothed_spike_train_ens2 = smooth_spike_train(spike_times, sampling_rate_ens2, duration = T_ens2,sigma_ms = filter_width)
smoothed_spike_train_cascade = smooth_spike_train(spike_times, sampling_rate_cascade, duration = T_cascade, sigma_ms = filter_width)
smoothed_prob_ens2 = smooth_prediction(spike_prob_ens2, sampling_rate_ens2, sigma_ms=filter_width)
smoothed_prob_cascade = smooth_prediction(spike_prob_cascade[0], sampling_rate_cascade, sigma_ms=filter_width)
smoothed_old_cas = smooth_prediction(spike_prob_ol_cas, sampling_rate_cascade, sigma_ms=filter_width)

# Setup plot
plt.figure(figsize=(12, 8))

# dff up top
plt.plot(time_stamps_ens2, dff, label='dff')
# ens2 with smoothing
plt.plot(time_stamps_ens2, spike_prob_ens2-1, label='spike_prob (ENS2)')
plt.plot(time_stamps_ens2, smoothed_prob_ens2-1, label='Smoothed ENS2', linestyle='--')
# cascade with smoothing
plt.plot(time_stamps_cascade[0], spike_prob_cascade[0]-2, label='spike_prob (Cascade)')
plt.plot(time_stamps_cascade[0], smoothed_prob_cascade-2, label='Smoothed Cascade', linestyle='--')
# smoothed old cascade
plt.plot(time_stamps_cascade[0], spike_prob_ol_cas[0]-3, label='spike_prob (Old Cascade)')
# smoothed spike train
plt.plot(time_stamps_ens2, smoothed_spike_train_ens2-4, color='k', label='Smoothed spike train')
# Add spikes as vlines
ymin, ymax = plt.gca().get_ylim()
plt.vlines(spike_times+time_stamps_ens2[0], ymin=ymin, ymax=ymax, color='k', alpha=0.5, label='Spikes')
plt.legend()


# 5. Compute correlations
corr_ens2, _, smoothed_prob_ens2 = compute_smoothed_correlation(spike_times, spike_prob_ens2, sampling_rate=sampling_rate_ens2,sigma_ms=filter_width)
corr_cascade, smoothed_spike_train_cascade, smoothed_prob_cascade = compute_smoothed_correlation(spike_times, spike_prob_cascade[0], sampling_rate=sampling_rate_cascade,sigma_ms=filter_width)
corr_ol_cas,_,_ = compute_smoothed_correlation(spike_times, spike_prob_ol_cas[0], sampling_rate=sampling_rate_cascade,sigma_ms=filter_width)

print(f"Correlation (ENS2): {corr_ens2}")
print(f"Correlation (Cascade): {corr_cascade}")
print(f"Correlation (Old Cascade): {corr_ol_cas}")

plt.tight_layout()
plt.show()
#plt.savefig("model_eval_plot.png", dpi=200, bbox_inches='tight')