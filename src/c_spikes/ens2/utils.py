# -*- coding: utf-8 -*-
"""
Utility functions for ENS2 integration: data loading, smoothing, early stopping
and weight initialization.
"""

import os
import numpy as np
import scipy.io as sio
from scipy import signal
from scipy.stats import invgauss
from scipy.ndimage import gaussian_filter

class EarlyStopping:
    """Early stops training if validation loss doesn't improve after a patience period."""
    def __init__(self, patience=7, verbose=False, delta=0.0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.verbose:
                    print("EarlyStopping: stopping early (no improvement).")
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        # Note: Could add model checkpoint saving here if needed.

def weights_init_normal(m):
    """Initialize convolution and linear layers with a normal distribution (mean=0, std=0.02)."""
    classname = m.__class__.__name__
    if "Conv" in classname:
        try:
            m.weight.data.normal_(0.0, 0.02)
        except AttributeError:
            pass
    elif "Linear" in classname:
        try:
            m.weight.data.normal_(0.0, 0.02)
        except AttributeError:
            pass

def build_causal_kernel(sampling_rate, smoothing):
    """Builds a causal smoothing kernel (inverse Gaussian) for spike train smoothing."""
    xx = np.arange(0, 199) / sampling_rate
    yy = invgauss.pdf(xx, smoothing/sampling_rate * 2.0, 101/sampling_rate, 1)
    ix = np.argmax(yy)
    yy = np.roll(yy, int((99 - ix) / 1.5))
    kernel = yy / np.nansum(yy)
    return kernel

def load_training_data(data_folder, sampling_rate, smoothing_std=0.025, use_causal_kernel=False):
    """
    Load training calcium and spike data from .mat files in the specified folder, and prepare training 
    segments for the model. Returns three arrays:
      - trace_segments: shape (N_segments, window_len) calcium snippets
      - rate_segments:  shape (N_segments, window_len) smoothed spike rate snippets (training targets)
      - spike_segments: shape (N_segments, window_len) binary spike snippets (for optional use with other losses)
      TODO: may want to make sampling rate a kwarg if pulling from a non-CASCADE compliant dataset
    """
    ephys_sampling_rate = 1e4  # spike event times in data are recorded at 10 kHz
    # Initialize accumulators (we'll determine window_len from first trial)
    trace_segments = np.empty((0, 0), dtype=np.float32)
    spike_segments = np.empty((0, 0), dtype=np.int_)
    rate_segments = np.empty((0, 0), dtype=np.float32)
    signal_len = None

    # Iterate over all .mat files in the training data folder
    for fname in sorted(os.listdir(data_folder)):
        if not fname.endswith('.mat'):
            continue
        mat = sio.loadmat(os.path.join(data_folder, fname))
        if 'CAttached' not in mat:
            continue

        # Each .mat contains 'CAttached': an array of trials for a given noise sample
        # following the CASCADE format
        for trial in mat['CAttached'][0]:
            trial_data = trial[0][0]

            # Ensure required fields exist in this trial's data
            try:
                fluo_times = np.squeeze(trial_data['fluo_time'])
                traces_mean = np.squeeze(trial_data['fluo_mean'])
                events = trial_data['events_AP']
            except Exception:
                # If any field is missing, skip this trial
                continue

            # Clean NaNs and empty events
            events = events[~np.isnan(events)]
            if events.size == 0:
                continue  # no spike events in this trial
            valid_idx = ~np.isnan(traces_mean)
            fluo_times = fluo_times[valid_idx]
            traces_mean = traces_mean[valid_idx]
            if traces_mean.size == 0:
                continue

            # Compute frame rate of this trial from time stamps
            frame_rate = 1 / np.mean(np.diff(fluo_times))

            # Restrict spike event times to within this trial's time span
            event_time = events / ephys_sampling_rate  # convert event indices to seconds
            event_time = event_time[(event_time >= fluo_times[0]) & (event_time <= fluo_times[-1])]
            if event_time.size == 0:
                continue

            # Resample fluorescence trace to the target sampling_rate
            num_samples = int(round(traces_mean.shape[0] * (sampling_rate / frame_rate)))
            traces_resampled, t_resampled = signal.resample(traces_mean, num_samples, t=fluo_times)
            frame_rate_resampled = 1 / np.nanmean(np.diff(t_resampled))

            # Align time to start at 0 for consistency
            t_resampled = t_resampled - t_resampled[0]
            event_time = event_time - t_resampled[0]

            # Filter events to resampled time range
            event_time = event_time[(event_time >= 0) & (event_time <= t_resampled[-1])]
            if event_time.size == 0 or traces_resampled.size == 0:
                continue
            # Determine segment window length (use 96 as default from ENS2 if not set)
            if signal_len is None:
                signal_len = 96  # default window length in samples (frames)
                trace_segments = np.zeros((0, signal_len), dtype=np.float32)
                spike_segments = np.zeros((0, signal_len), dtype=np.int_)
                rate_segments = np.zeros((0, signal_len), dtype=np.float32)
            # Pad trace and event arrays at the beginning and end by half the window length
            pad = signal_len // 2
            padded_trace = np.concatenate([np.zeros(pad), traces_resampled, np.zeros(pad)])

            # Bin spike times into the resampled time bins
            bin_edges = np.append(t_resampled, t_resampled[-1] + 1/frame_rate_resampled)
            events_binned, _ = np.histogram(event_time, bins=bin_edges)
            events_binned = np.concatenate([np.zeros(pad, dtype=int), events_binned, np.zeros(pad, dtype=int)])

            # Smooth the binned spike train to create a continuous rate target
            if use_causal_kernel:
                smoothing = smoothing_std * sampling_rate  # effective window (in frames) for smoothing
                kernel = build_causal_kernel(sampling_rate, smoothing)
                events_binned_smooth = np.convolve(events_binned.astype(float), kernel, mode='same')
            else:
                sigma = smoothing_std * sampling_rate  # smoothing in frames for Gaussian
                events_binned_smooth = gaussian_filter(events_binned.astype(float), sigma=sigma)

            # Slide a window of length signal_len across the padded trace to generate segments
            total_len = len(events_binned)  # length of padded series
            data_len = total_len - signal_len + 1
            if data_len <= 0:
                continue  # skip if trial is too short for one segment
            X = np.zeros((data_len, signal_len), dtype=np.float32)
            Y_spike = np.zeros((data_len, signal_len), dtype=np.int_)
            Y_rate = np.zeros((data_len, signal_len), dtype=np.float32)
            for t in range(data_len):
                X[t, :] = padded_trace[t:t+signal_len]
                Y_spike[t, :] = events_binned[t:t+signal_len]
                Y_rate[t, :] = events_binned_smooth[t:t+signal_len]

            if np.isnan(X).any():
                pass
            # Append this trial's segments to the global list
            trace_segments = np.vstack([trace_segments, X])
            spike_segments = np.vstack([spike_segments, Y_spike])
            rate_segments = np.vstack([rate_segments, Y_rate])
    return trace_segments, rate_segments, spike_segments
