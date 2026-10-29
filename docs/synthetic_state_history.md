# Inspect synthetic state concentrations and fluorescence

This recipe uses the existing public CPU backend and primary history API. It
requires a built CPU extension and the package dependencies; qualification used
Python 3.10. Set the backend **before importing** the package (restart an existing
Python session if it already imported a different backend).

```python
import os
os.environ["C_SPIKES_PGAS_BACKEND"] = "cpu"

from importlib.resources import files, as_file
import numpy as np
from c_spikes.pgas import pgas_bound as pgas
from c_spikes.syn_gen import simulate_state_history

# Packaged indicator kinetics; load the raw values for each new model.
with as_file(files("c_spikes.pgas").joinpath("20230525_gold.dat")) as parameter_file:
    gparams = np.loadtxt(parameter_file)

# Existing native order: G_tot, gamma, DCaT, Rf, gam_in, gam_out.
# These explicit example values are not universal recommendations for a cell.
cparams = np.array([6.04700454e-5, 1000.0, 1e-5, 5.0, 40.0, 4.0])
model = pgas.GCaMP(gparams, cparams)
model.setTimeStepMode(pgas.TimeStepMode.FIXEDLA)

time_s = np.array([0.0, 0.00005, 0.0001, 0.0002, 0.001])
spike_times_s = np.array([], dtype=np.float64)
history = simulate_state_history(model, time_s, spike_times_s, reset=True)
dff = model.getDFFValues()  # separate owned, dimensionless fluorescence array

# history.time_s: owned float64 array, shape (n,), seconds
# history.state_M: owned contiguous float64 array, shape (n, 12), molar
# dff: owned float64 array, shape (n,), corresponding to history.time_s
```

Columns are `G0` through `G8`, `BCa`, `Ca`, `Ca_in`: native zero-based indicator
state concentrations, endogenous bound calcium, free calcium, and the internal
pool variable. These are concentrations in M, not twelve normalized probabilities.
The helper does not return fluorescence; retrieve it separately as shown above.
The first fluorescence sample uses the actual initial indicator state and existing
calibration. A resting example can start at zero; a nonresting state need not.

To inspect an explicitly chosen nonresting state through existing APIs:

```python
model.setState(1e-7)  # existing equilibrium/state setup at this free calcium, in M
history = simulate_state_history(model, time_s, spike_times_s, reset=False)
dff = model.getDFFValues()
```

`reset=True` calls the existing `model.init()` before integration. `reset=False`
uses the stored state. Both overwrite native histories and modify existing rate
and matrix caches. Integration does not write its final evolved state back into
the stored model state: repeated calls are **not chained continuation**. The
helper offers no rollback guarantee. Returned arrays remain independent of later
model calls and supplied arrays; changing them does not edit the native histories.

Time samples must be finite, one-dimensional, strictly increasing and contain at
least two elements. Spike times must be finite, one-dimensional and sorted.
Inputs are copied to float64. Absolute timestamps, duplicate spikes and spikes
outside the window are passed through without clipping or retiming. The primary
method extends its existing internal grid as needed to cover the requested end;
it does not rebase the time origin or clamp/extrapolate the requested samples.

Metadata records `buffer_approximation="linear"`,
`integration_method="integrateOverTime"`, and `internal_dt_s=1e-4`. This primary
history method uses a linear endogenous-buffer approximation and explicit
100-microsecond internal steps. The separate evolution-mode setting does not
dispatch this history method; setting FIXEDLA above makes the intended model
configuration explicit without claiming a different history implementation.

Input validation errors occur before model access/mutation. Nonfinite numerical
results remain visible with existing diagnostics; they are not filtered,
normalized or converted into new numerical exceptions. Finite output alone does
not establish scientific accuracy, stability or a valid parameter regime.

If you already have a `synth_gen` instance, its existing `.gcamp` model can be
supplied explicitly. This does not synchronize the generator object's stored
cell-parameter configuration automatically. You do not need to construct a
`synth_gen` instance merely to inspect histories: that constructor also enumerates
noise files, including when noise is disabled. This standalone recipe avoids
those dataset/noise setup requirements through the existing native model API.

The history helper does not generate spikes/noise, draw random numbers, write
files or change dataset generation. Focused CPU checks cover reset, ownership,
input boundaries, endpoint coverage and initial fluorescence. They do not qualify
all installed wheels, GPU paths, broad parameter domains or full inference.

`simulate_state_occupancies` returns the raw history together with independent
indicator and buffer fractions, captured pool totals and denominator-validity
metadata. Indicator states use the actual G_tot pool; bound buffer uses B_tot.
Invalid denominators produce invalid fraction metadata without clipping,
renormalizing or altering the raw concentrations. Native `getPoolTotals()` exposes
the current pools; stale constructor parameters are not authoritative totals.

Analyzer append is a validated parameter warm restart, not an exact RNG/chain
checkpoint. It validates the eleven-column parameter record and companion output
structure before writing. `niter` is the target total number of saved records;
`get_run_status()` and `get_run_message()` distinguish completion, no-op and
invalid restart. Completion does not certify numerical or scientific validity.

## Retired alternative history API

`GCaMP.integrateOverTime2` has been removed from the native and Python APIs. It
used different calcium equations and did not publish matching concentration histories.
There is no compatibility alias: substituting `integrateOverTime` silently would change
the scientific calculation. Review the primary `simulate_state_history` recipe above
when migrating. Historical source remains in Git; this retirement does not establish
that the alternative equations were scientifically incorrect.
