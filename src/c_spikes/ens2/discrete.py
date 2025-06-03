# Utility for converting rate to discrete spike train and extracting events.

import copy
import numpy as np
from scipy.ndimage.filters import gaussian_filter

def estimate_spike(rate, sampling_rate, smoothing_std, debug=False):
    """
    From ENS2.estimate_spike:
    Takes a continuous spike‐rate array and returns a binary spike train.
    
    Args:
        rate (array‐like): continuous ENS2 output per frame
        sampling_rate (float): frames per second
        smoothing_std (float): std (in seconds) used for smoothing in ENS2
        debug (bool): if True, uses the alternate aggressive threshold suggested by authors
            with Gaussian smoothing applied prior to thresholding.

    Returns:
        est_spike (np.ndarray): float32 array of the same shape as `rate`,
                                with 1.0 at inferred spike frames, 0.0 elsewhere.
    """
    # copy & squeeze
    r = np.float32(np.array(copy.deepcopy(rate))).squeeze()
    std = smoothing_std

    # threshold out background “bubbles” - author's term for noise
    if not debug:
        r[r < 0.02/std] = 0
    else:
        mask = r > 0.02/std
        if mask.any():
            r[r < np.sqrt(np.mean(r[mask]))] = 0

    # find onsets/offsets
    rate_diff = np.diff(np.int8(r > 0))
    est_spike = np.zeros_like(r, dtype='float32')
    onset = 0

    for idx in range(len(rate_diff)):
        if rate_diff[idx] == 1:
            # start of a nonzero segment
            onset = idx + 1

        elif rate_diff[idx] == -1 and onset > 0:
            # end of a nonzero segment
            offset = idx
            slices = r[onset:offset+1]

            # initialize single‐spike guess
            could_add = True
            cur_spike = np.zeros_like(slices, dtype='float32')
            if slices.sum() >= 0.5:
                cur_spike[np.argmax(slices)] = 1

            # smooth that guess
            cur_rate = gaussian_filter(cur_spike, sigma=std, mode='constant', cval=0.)
            cur_loss = np.sum((slices - cur_rate)**2)

            # iteratively add spikes if they reduce MSE
            while could_add:
                # candidate matrix: each row adds one spike at a new location
                cand_spikes = cur_spike + np.eye(len(slices), dtype='float32')
                # note: original used sigma=(0,std), we do the same
                cand_rates = gaussian_filter(cand_spikes, sigma=(0, std), mode='constant', cval=0.)
                # compute per‐row loss
                losses = np.sum((slices - cand_rates)**2, axis=1)
                new_loss = losses.min()
                argmin = int(losses.argmin())

                if new_loss - cur_loss <= -1e-8:
                    # accept the best new spike
                    cur_spike = cand_spikes[argmin]
                    cur_rate = cand_rates[argmin]
                    cur_loss = new_loss
                else:
                    # no further improvement: write out this segment
                    est_spike[onset:offset+1] = cur_spike
                    could_add = False

        # guard against runaway segments >500
        elif idx - onset >= 499:
            if idx+1 < len(rate_diff):
                rate_diff[idx+1] = -1
            if idx+2 < len(rate_diff):
                rate_diff[idx+2] = 1

    return est_spike


def extract_event(spike_train, sampling_rate):
    """
    From ENS2.extract_event:
    Converts a binary‐valued spike train into a sorted list of event times (s).

    Args:
        spike_train (array‐like): output of estimate_spike (0.0 or 1.0)
        sampling_rate (float): frames per second

    Returns:
        events (list of float): spike times in seconds, sorted ascending
    """
    st = np.squeeze(copy.deepcopy(spike_train))
    events = []

    # each pass removes one “layer” of overlapping spikes until none remain
    while (st > 0).any():
        idxs = np.where(st > 0)[0]
        # +1 to mirror original (they did idx+1)
        events += ((idxs + 1) / sampling_rate).tolist()
        st = st - 1  # peel off one layer

    events.sort()
    return events