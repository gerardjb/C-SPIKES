# General utility functions for the c_spikes package
import scipy.io as sio
import numpy as np

def load_Janelia_data(j_path):
    """
    Load Janelia dataset from a .mat file. This is for the data as in the custom
    imports done for Broussard et al. 2025 of Marton Rosza's Janelia dandi dataset
    released with the Zhang et al. 2023 paper describing the jGCaMP8 sensors.
    Args:
        j_path (str): Path to the .mat file containing Janelia dataset.
    Returns:
        tuple: Contains time_stamps, dff (fluorescence data), and spike_times.
    """

    all_data = sio.loadmat(j_path)
    dff = all_data['dff']
    time_stamps = all_data['time_stamps']
    spike_times = all_data['ap_times'] 

    return time_stamps, dff, spike_times

def spike_times_2_binary(spike_times,time_stamps):
    """
    Convert spike times to a binary vector based on provided time stamps.
    Args:
        spike_times (np.ndarray): Array of spike event times.
        time_stamps (np.ndarray): Array of time stamps to create binary vector against.
    Returns:
        np.ndarray: Binary vector where each element corresponds to a time stamp.
    """
    # initialize the binary vector
    binary_vector = np.zeros(len(time_stamps), dtype=int)

    # get event times within the time_stamps ends
    good_spike_times = spike_times[(spike_times >= time_stamps[0]) & (spike_times <= time_stamps[-1])]
    
    # Find the nearest element in 'a' that is less than the elements in 'b'
    for event_time in good_spike_times:
        # Find indices where 'a' is less than 'event_time'
        valid_indices = np.where(time_stamps < event_time)[0]
        if valid_indices.size > 0:
            nearest_index = valid_indices[-1]  # Taking the last valid index
            binary_vector[nearest_index] += 1

    return binary_vector

def unroll_pgas_traj(dat_file):
    """
    Convert spike times to a binary vector based on provided time stamps.
    Args:
        dat_file (str): Path to the .csv file containing PGAS trajectory data.
    Returns:
        index (np.ndarray): PGAS trajectory index / time-bin index.
        burst (np.ndarray): Burst state (discrete, 0 or 1).
        B (np.ndarray): Baseline drift (Brownian motion).
        S (np.ndarray): Discretized spike number per time bin.
        C (np.ndarray): "Calcium" value, akin to a DFF-like metric.
        Y (np.ndarray): Original data (not included in PGBAR output, but present in PGAS).
    """
    #Loading data
    data = np.genfromtxt(dat_file, delimiter=',', skip_header=1)
    #Dealing out data
    index = data[:,0]
    burst = data[:,1]
    B = data[:,2]
    S = data[:,3]
    C = data[:,4]
    
    #Note that files produced by PGBAR (rather than the more general PGAS) lack Y - for now not including it
    try:
        Y = data[:,5]
    except:
        Y = np.nan
    
    return index,burst,B,S,C,Y

def unroll_mean_pgas_traj(dat_file, logprob_file, burnin=100):
    """
    Read in and upack a PGAS trajectory file and take the mean trajectory across post-burnin. Spike
    train is returned as MAP.
    Args:
        dat_file (str): Path to the .csv file containing PGAS trajectory data.
        logprob_file (str): Path to the .csv file containing log-probabilities for each trajectory.
        burnin (int): Number of initial samples to discard as burn-in.
    Returns:
        baseline_mean (np.ndarray): Baseline drift (Brownian motion).
        burst_mean (np.ndarray): Burst state (discrete, 0 or 1).
        spikes_mean (np.ndarray): Discretized spike number per time bin.
        C_mean (np.ndarray): "Calcium" value, akin to DFF.
    """
    # Loading data
    data = np.genfromtxt(dat_file, delimiter=",", skip_header=1)
    if data.ndim == 1:
        data = np.asarray([data], dtype=float)
    logprob = np.atleast_1d(np.genfromtxt(logprob_file))
    logprob = np.asarray(logprob, dtype=float).ravel()
    if logprob.size == 0:
        raise ValueError(f"Empty logprob file: {logprob_file}")

    # Dealing out data
    index = data[:, 0]
    burst = data[:, 1]
    B = data[:, 2]
    S = data[:, 3]
    C = data[:, 4]

    # The trajectory dump is organized as:
    #   blocks of length TIME for each MCMC/trajectory sample (index == sample_id).
    # Therefore:
    #   TIME = count(index == 0)
    #   n_samples = number of samples (typically == niter)
    # and each series reshapes to (n_samples, TIME).
    TIME = int(np.sum(index == 0))
    if TIME <= 0:
        raise ValueError(f"PGAS traj file has no TIME axis (no index==0 rows): {dat_file}")

    n_samples_by_data = int(data.shape[0] // TIME)
    if n_samples_by_data <= 0:
        raise ValueError(f"PGAS traj file too short (TIME={TIME}): {dat_file}")
    if n_samples_by_data * TIME != data.shape[0]:
        raise ValueError(
            f"PGAS traj file size {data.shape[0]} is not divisible by TIME={TIME}: {dat_file}"
        )

    # If the logprob length doesn't match, truncate to the shortest so MAP selection is valid.
    n_samples = n_samples_by_data
    if logprob.size:
        n_samples = min(n_samples, int(logprob.size))
        logprob = logprob[:n_samples]
    if n_samples <= 0:
        raise ValueError(f"PGAS has no usable samples for mean/MAP: {dat_file}")

    # Truncate data arrays if needed to match n_samples.
    n_rows = n_samples * TIME
    burst = burst[:n_rows]
    B = B[:n_rows]
    S = S[:n_rows]
    C = C[:n_rows]

    S_mat = S.reshape((n_samples, TIME))
    burst_mat = burst.reshape((n_samples, TIME))
    B_mat = B.reshape((n_samples, TIME))
    C_mat = C.reshape((n_samples, TIME))

    burnin_eff = int(max(0, min(int(burnin), n_samples - 1)))
    sample_slice = slice(burnin_eff, None)

    spikes_mean = np.mean(S_mat[sample_slice, :], axis=0)
    burst_mean = np.mean(burst_mat[sample_slice, :], axis=0)
    baseline_mean = np.mean(B_mat[sample_slice, :], axis=0)
    C_mean = np.mean(C_mat[sample_slice, :], axis=0)
    
    # MAP spike train
    max_logprob_ind = n_samples - 1
    if logprob.size:
        post = logprob[sample_slice]
        if post.size:
            max_logprob_ind = int(np.argmax(post)) + burnin_eff
    spikes_MAP = S_mat[max_logprob_ind, :]

    return burst_mean,baseline_mean,spikes_mean,C_mean,spikes_MAP
