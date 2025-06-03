# High-level training and inference functions for ENS2 model integration

import os
import numpy as np
import torch
import torch.optim as optim
from tqdm.auto import trange
from c_spikes.ens2.utils import EarlyStopping, weights_init_normal, load_training_data
from c_spikes.ens2.models import UNet

def train_model(model_name, sampling_rate=60, smoothing_std=0.025, use_causal_kernel=False,
                epochs=5000, patience=500, batch_size=1024, learning_rate=0.001, loss_type='MSE', verbose=0):
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
    data_folder = os.path.join('results/Ground_truth', f'synth_{model_name}')
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
    # (spike_tensor is prepared in case alternative losses use it)
    spike_tensor = torch.FloatTensor(spike_segments)
    N = trace_tensor.shape[0]
    # Initialize model and optimizer
    model = UNet()  # default: 96-length window, init_features=9 (≈150K params)
    model.apply(weights_init_normal)  # random weight init
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    # One of these days I might put in other metrics from the paper
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
    save_dir = os.path.join('Pretrained_models', model_name)
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, 'ens2_model.pth'))
    if verbose:
        print(f"ENS2 model saved to {save_dir}/ens2_model.pth")

def predict(model_name, traces, sampling_rate, smoothing_std=0.025):
    """
    Use a trained ENS2 model to predict spike activity for given fluorescence traces.
    Parameters:
      model_name (str): Name of the trained model to load.
      traces (array-like): 1D array of fluorescence (ΔF/F) values (single neuron trace).
      sampling_rate (float): Frame rate of the data (Hz).
      smoothing_std (float): Smoothing std (s) used in training (for thresholding logic).
    Returns:
      predicted_rate (np.ndarray): Predicted continuous spike probability/rate for each timepoint.
    """
    # Load the trained model weights
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UNet()
    model_path = os.path.join('Pretrained_models', model_name, 'ens2_model.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    # Prepare input trace
    trace = np.asarray(traces).squeeze()
    T = trace.shape[0]
    window_len = 96  # model window length
    pad = window_len // 2
    # Pad trace at both ends
    padded = np.concatenate([np.zeros(pad), trace, np.zeros(pad)])
    # Allocate accumulation buffer for output
    accum_output = np.zeros(T + window_len - 1, dtype=np.float32)
    # Slide window across trace and accumulate model predictions
    for i in range(T):
        segment = padded[i : i + window_len]
        seg_tensor = torch.from_numpy(segment.astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_segment = model(seg_tensor).cpu().numpy()  # shape (window_len,)
        accum_output[i : i + window_len] += pred_segment.astype(np.float32)
    # Average the overlapping predictions
    predicted_rate = accum_output[pad : pad + T] / window_len
    return predicted_rate
