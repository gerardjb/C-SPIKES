# High-level training and inference functions for ENS2 model integration

import os
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import trange
from c_spikes.ens2.utils import EarlyStopping, weights_init_normal, load_training_data
from c_spikes.ens2.models import UNet

def train_model(model_name, tag="", sampling_rate=60, smoothing_std=0.025, use_causal_kernel=False,
                epochs=5000, patience=500, batch_size=512, learning_rate=0.001, loss_type='MSE', verbose=0):
    """
    Train an ENS2 neural network for spike inference on synthetic data.
    Parameters:
      model_name (str): Name for the model (used for saving outputs).
      sampling_rate (float): Frame rate (Hz) of the training data.
      smoothing_std (float): Gaussian smoothing sigma (in seconds) for ground-truth spikes.
      use_causal_kernel (bool): If True, use a causal kernel instead of Gaussian for smoothing.
      epochs (int): Maximum training epochs.
      patience (int): Patience for early stopping (no improvement epochs).
      batch_size (int): Batch size for training (randomly sampled each epoch).
      learning_rate (float): Learning rate for Adam optimizer.
      loss_type (str): Loss function ('MSE' supported; placeholders for 'EucD', 'Corr').
      verbose (int): If >0, print progress and info during training.
    """
    # Load synthetic (or whatever) training data
    data_folder = os.path.join('results/Ground_truth', f'synth_{tag}')
    trace_segments, rate_segments, spike_segments = load_training_data(
        data_folder, sampling_rate, smoothing_std=smoothing_std, use_causal_kernel=use_causal_kernel
    )
    assert not np.isnan(trace_segments).any() and not np.isinf(trace_segments).any(), \
        "NaN or Inf in trace_segments!"
    assert not np.isnan(rate_segments).any()  and not np.isinf(rate_segments).any(), \
        "NaN or Inf in rate_segments!"
    if trace_segments.size == 0:
        raise FileNotFoundError(f"No training data found in {data_folder}")
    # Convert to torch tensors
    trace_tensor = torch.FloatTensor(trace_segments)
    rate_tensor  = torch.FloatTensor(rate_segments)
    N = trace_tensor.shape[0]

    # Initialize model and optimizer
    model = UNet()  # default: 96-length window, init_features=9 (≈150K params)
    model.apply(weights_init_normal)  # random weight init
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # One of these days I might put in other metrics from the paper; they did suggest those suck, though
    if loss_type != 'MSE':
        print(f"Warning: Loss type '{loss_type}' not implemented. Using MSE.")
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Setup early stopping
    stopper = EarlyStopping(patience=patience, verbose=(verbose > 0))
    # Training loop
    for epoch in (trange(1, epochs+1) if verbose else range(1, epochs+1)):
        model.train()
        # Sample a random batch from all segments
        idx = torch.randint(0, N, (min(batch_size, N),))
        batch_trace = trace_tensor[idx].to(device)
        batch_rate  = rate_tensor[idx].to(device)
        optimizer.zero_grad()
        outputs = model(batch_trace)
        loss = criterion(outputs, batch_rate)
        loss.backward()
        optimizer.step()
        # Early stopping check on training loss
        stopper(loss.item(), model)
        if stopper.early_stop:
            if verbose:
                print(f"Early stopping triggered at epoch {epoch}.")
            break
    # Save trained model weights
    save_dir = os.path.join('results/Pretrained_models', f"ens_{model_name}")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, 'ens2_model.pth'))
    if verbose:
        print(f"ENS2 model saved to {save_dir}/ens2_model.pth")

def predict(model_name, traces, sampling_rate, smoothing_std=0.05, model_folder='results/Pretrained_models',window_len=96, batch_size=8192):
    """
    Use a trained ENS2 model to predict spike activity for given fluorescence traces.
    Parameters:
      model_name (str): Name of the trained model to load.
      traces (array-like): 1 or 2D array of fluorescence (ΔF/F) values (single neuron trace) size (n_trace x time).
      sampling_rate (float): Frame rate of the data (Hz).
      smoothing_std (float): Smoothing std (s) used in training (for thresholding logic).
      model_folder (str): Folder where the trained model weights are stored.
      window_len (int): Length of the input window for the model (default 96).
      batch_size (int): Batch size for processing input traces (default 8192 as ENS2).
    Returns:
      predicted_rate (np.ndarray): Predicted continuous spike probability/rate for each timepoint.
    """
    # Load the trained model weights to target device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UNet()
    model_path = os.path.join(model_folder, f"ens_{model_name}", 'ens2_model.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Prepare input trace
    traces = np.asarray(traces).squeeze()
    if traces.ndim == 1:
        traces = traces[np.newaxis, :]  # Convert to 2D for consistency
    n_traces, T = traces.shape
    window_len = 96  # model window length
    pad = window_len // 2

    # Allocate output array for predicted rates
    predicted_rates = np.zeros((n_traces, T), dtype=np.float32)

    start_time = time.perf_counter()
    # Process each trace independently
    for i_trace in range(n_traces):
        trace = traces[i_trace]
        padded = np.concatenate([np.zeros(pad), trace, np.zeros(pad)])

        # 
        padded_tensor = torch.from_numpy(padded).float()  # shape (T+2*pad,)
        # Use unfold to get a (T, window_len) tensor of all length-96 windows
        # unfold(dim=0, size=window_len, step=1)
        windows = padded_tensor.unfold(0, window_len, 1)[:-1]  # shape (T, 96)

        # Build a 2D array of shape (T, window_len), where each row is a window
        #windows = np.lib.stride_tricks.sliding_window_view(padded, window_shape=window_len)
        #windows = windows[:T] # only need first T rows

        # Convert to torch and wrap in DataLoader
        #tensor_windows = torch.from_numpy(windows).float()  # shape (T,96)
        ds = TensorDataset(windows)                   # each sample: (96,)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

        # Storage for all batch outputs: shape (T, window_len)
        all_out = np.zeros((T, window_len), dtype=np.float32)
        start_idx = 0  # to track where this batch’s output goes

        # Process the batches
        with torch.no_grad():
            for batch in dl:
                batch_tensors = batch[0].to(device)  # shape (bs, 96)
                out_batch = model(batch_tensors).cpu().numpy()  # (bs, 96)
                bs = out_batch.shape[0]
                all_out[start_idx : start_idx + bs, :] = out_batch
                start_idx += bs

        # Reconstruct continuous prediction by summing overlapping windows
        accum = np.zeros(T + window_len - 1, dtype=np.float32)
        for align_idx in range(T):
            accum[align_idx : align_idx + window_len] += all_out[align_idx]

        # Extract centered portion and normalize
        predicted_rates[i_trace] = accum[pad : pad + T] / window_len

    if predicted_rates.shape[0] == 1:
        return predicted_rates[0]  # Return 1D if input was 1D

    stop_time = time.perf_counter()
    print(f"time to load and process tensors was {stop_time - start_time}")
    return predicted_rates
