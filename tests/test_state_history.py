"""Owned public history contracts, independent of a frozen provider oracle."""
import numpy as np
import pytest
from c_spikes.syn_gen import simulate_state_history, simulate_state_occupancies


class Model:
    def __init__(self):
        self.calls = []; self.raw = np.arange(24, dtype=float).reshape(2, 12)
        self.totals = {"G_tot_M": 2., "B_tot_M": 4.}
    def init(self): self.calls.append("init")
    def integrateOverTime(self, time, spikes):
        self.calls.append((time.copy(), spikes.copy()))
    def getStates(self):
        return {"G": self.raw[:, :9], "BCa": self.raw[:, 9], "Ca": self.raw[:, 10], "Ca_in": self.raw[:, 11]}
    def getPoolTotals(self): return self.totals.copy()


def test_raw_history_owns_values_preserves_nonfinite_and_reset_contract():
    m = Model(); m.raw[0, 0] = np.nan; m.raw[1, 11] = np.inf
    time = np.array([1., 1.001]); spikes = np.array([0., 1., 1., 2.])
    out = simulate_state_history(m, time, spikes)
    assert m.calls[0] == "init"
    np.testing.assert_array_equal(m.calls[1][0], time)
    np.testing.assert_array_equal(m.calls[1][1], spikes)
    np.testing.assert_array_equal(out.state_M, m.raw)
    assert out.state_M.flags.owndata and out.time_s.flags.owndata
    assert not np.shares_memory(out.state_M, m.raw) and not np.shares_memory(out.time_s, time)
    assert out.state_order == tuple(f"G{i}" for i in range(9)) + ("BCa", "Ca", "Ca_in")
    assert out.internal_dt_s == 1e-4 and out.buffer_approximation == "linear"
    m.calls.clear(); simulate_state_history(m, time, spikes, reset=False)
    assert len(m.calls) == 1


@pytest.mark.parametrize("time,spikes", [([], []), ([0.], []), ([1., 0.], []), ([0., 0.], []), ([0., np.nan], []), ([[0., 1.]], []), ([0., 1.], [1., 0.]), ([0., 1.], [np.inf])])
def test_input_validation_precedes_model_mutation(time, spikes):
    m = Model()
    with pytest.raises(ValueError): simulate_state_history(m, time, spikes)
    assert m.calls == []


def test_occupancies_use_separate_pools_without_clipping_or_renormalization():
    m = Model(); m.raw[0, 0] = -1.
    out = simulate_state_occupancies(m, [0., 1.], [])
    np.testing.assert_array_equal(out.indicator_fraction, m.raw[:, :9] / 2.)
    np.testing.assert_array_equal(out.buffer_bound_fraction, m.raw[:, 9] / 4.)
    np.testing.assert_array_equal(out.history.state_M, m.raw)
    assert out.valid_denominators == (True, True) and out.pool_totals_M == (2., 4.)
    assert not np.shares_memory(out.indicator_fraction, out.history.state_M)
    assert not np.shares_memory(out.buffer_bound_fraction, out.history.state_M)


@pytest.mark.parametrize("total", [0., -1., np.nan, np.inf])
def test_invalid_pool_metadata_preserves_raw_values(total, caplog):
    m = Model(); m.totals['G_tot_M'] = total
    out = simulate_state_occupancies(m, [0., 1.], [])
    assert out.valid_denominators == (False, True) and np.isnan(out.indicator_fraction).all()
    np.testing.assert_array_equal(out.buffer_bound_fraction, m.raw[:, 9] / 4.)
    np.testing.assert_array_equal(out.history.state_M, m.raw)
    assert len(caplog.records) == 1 and "denominator invalid" in caplog.records[0].message
