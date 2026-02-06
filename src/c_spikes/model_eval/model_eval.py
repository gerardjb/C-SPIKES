import numpy as np
from scipy.ndimage import gaussian_filter1d

def smooth_spike_train(spike_times, sampling_rate, duration=None, sigma_ms=50, binning="linear"):
    """
    Bin spike times into a high-resolution array and apply Gaussian smoothing.
    Args:
        spike_times (array-like): Spike event times in seconds.
        sampling_rate (float): Sampling rate in Hz for the binned spike train.
        duration (float, optional): Total duration in seconds. If None, uses max(spike_times).
        sigma_ms (float): Standard deviation of Gaussian kernel in milliseconds.
        binning (str): How to place spikes on the discrete grid before smoothing.
            - "linear" (default): distribute each spike between the two neighboring bins
              based on its fractional index (reduces timing-quantization error at low Fs).
            - "round": assign each spike to the nearest bin.
            - "floor": assign each spike to the left bin (legacy behavior).
    Returns:
        np.ndarray: Smoothed spike-rate array (length = duration * sampling_rate).
    """
    spike_times = np.asarray(spike_times)
    if duration is None:
        duration = spike_times.max() if spike_times.size > 0 else 0
    N = int(np.ceil(duration * sampling_rate)) + 1
    spike_array = np.zeros(N, dtype=float)
    pos = spike_times * float(sampling_rate)
    if binning == "floor":
        indices = pos.astype(int)
        indices = indices[(indices >= 0) & (indices < N)]
        np.add.at(spike_array, indices, 1.0)
    elif binning == "round":
        indices = np.rint(pos).astype(int)
        indices = indices[(indices >= 0) & (indices < N)]
        np.add.at(spike_array, indices, 1.0)
    elif binning == "linear":
        i0 = np.floor(pos).astype(int)
        frac = pos - i0
        i1 = i0 + 1

        m0 = (i0 >= 0) & (i0 < N)
        if np.any(m0):
            np.add.at(spike_array, i0[m0], (1.0 - frac[m0]).astype(float))
        m1 = (i1 >= 0) & (i1 < N)
        if np.any(m1):
            np.add.at(spike_array, i1[m1], frac[m1].astype(float))
    else:
        raise ValueError(f"Unsupported binning mode: {binning!r}. Use 'linear', 'round', or 'floor'.")
    sigma_samples = (sigma_ms / 1000) * sampling_rate
    smoothed = gaussian_filter1d(spike_array, sigma=sigma_samples)

    return smoothed

def smooth_prediction(prediction_trace, sampling_rate, sigma_ms=50):
    """
    Apply Gaussian smoothing to a continuous prediction trace.
    Args:
        prediction_trace (array-like): Continuous predictions (e.g., spike_prob).
        sampling_rate (float): Sampling rate in Hz of the prediction_trace.
        sigma_ms (float): Standard deviation of Gaussian kernel in milliseconds.
    Returns:
        np.ndarray: Smoothed prediction trace.
    """
    pred = np.asarray(prediction_trace, dtype=float)
    sigma_samples = (sigma_ms / 1000) * sampling_rate
    return gaussian_filter1d(pred, sigma=sigma_samples)

def compute_smoothed_correlation(spike_times, prediction_trace, sampling_rate, sigma_ms=50):
    """
    Smooth both spike train and prediction, then compute their Pearson correlation.
    Args:
        spike_times (array-like): Spike event times in seconds.
        prediction_trace (array-like): Continuous predictions aligned with spike train.
        sampling_rate (float): Sampling rate in Hz.
        sigma_ms (float): Gaussian kernel width in milliseconds.
    Returns:
        corr (float): Pearson correlation coefficient between smoothed signals.
        spikes_smooth (np.ndarray): Smoothed spike-rate array.
        pred_smooth (np.ndarray): Smoothed prediction trace.
    """
    # Determine duration based on prediction length if not set by spike times
    duration = (len(prediction_trace)-1) / sampling_rate
    spikes_smooth = smooth_spike_train(spike_times, sampling_rate, duration, sigma_ms)
    pred_smooth = smooth_prediction(prediction_trace, sampling_rate, sigma_ms)
    # Align lengths
    min_len = min(spikes_smooth.size, pred_smooth.size)
    spikes_smooth = spikes_smooth[:min_len]
    pred_smooth = pred_smooth[:min_len]
    # Pearson correlation
    x = spikes_smooth
    y = pred_smooth
    x_mean, y_mean = np.nanmean(x), np.nanmean(y)
    cov = np.nanmean((x - x_mean) * (y - y_mean))
    corr = cov / (np.nanstd(x) * np.nanstd(y))
    return corr, spikes_smooth, pred_smooth

# Example usage:
# spike_times = [0.1, 0.35, 0.4, 1.2]  # in seconds
# pred = np.random.rand(5000)          # predicted_rate at 100 Hz for 50 s
# sr = 100                             # Hz
# corr, s_spike, s_pred = compute_smoothed_correlation(spike_times, pred, sr, sigma_ms=50)
# print("Smoothed correlation:", corr)
