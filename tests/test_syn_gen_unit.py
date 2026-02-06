from c_spikes.syn_gen.synth_gen import synth_gen


class DummyGCaMP:
    def __init__(self):
        self.params = None
        self.init_called = False

    def setParams(self, *args):
        self.params = args

    def init(self):
        self.init_called = True


def _seed_noise_dir(tmp_path, count: int = 3):
    for idx in range(count):
        (tmp_path / f"noise_{idx}.mat").write_bytes(b"0")
    return tmp_path


def test_syn_gen_accepts_stub_gcamp(tmp_path):
    noise_dir = _seed_noise_dir(tmp_path, count=2)
    dummy = DummyGCaMP()
    gen = synth_gen(noise_dir=str(noise_dir), GCaMP_model=dummy, use_noise=False)

    assert gen.gcamp is dummy
    assert gen.noise_dir == str(noise_dir)


def test_syn_gen_noise_subset_reproducible(tmp_path):
    noise_dir = _seed_noise_dir(tmp_path, count=4)
    gen = synth_gen(noise_dir=str(noise_dir), GCaMP_model=DummyGCaMP(), noise_fraction=0.5)

    subset_a = gen._select_noise_subset(seed=123)
    subset_b = gen._select_noise_subset(seed=123)

    assert subset_a == subset_b
    assert len(subset_a) == 2
