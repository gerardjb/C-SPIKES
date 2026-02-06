from pathlib import Path
from typing import Dict

import numpy as np
from scipy.io import loadmat


def extract_time_stamp_edges(data_dir: Path) -> Dict[str, np.ndarray]:
    """
    Load every .mat file in data_dir, grab the first and last element from each
    row of the time_stamps array, and return a dictionary keyed by filename.
    """
    results: Dict[str, np.ndarray] = {}
    for mat_path in sorted(data_dir.glob("*.mat")):
        contents = loadmat(mat_path)
        if "time_stamps" not in contents:
            raise KeyError(f"'time_stamps' field missing in {mat_path}")

        time_stamps = np.asarray(contents["time_stamps"])
        if time_stamps.ndim != 2 or time_stamps.shape[1] < 1:
            raise ValueError(f"Unexpected shape for time_stamps in {mat_path}: {time_stamps.shape}")

        first_last = np.column_stack((time_stamps[:, 0], time_stamps[:, -1]))
        key = mat_path.stem
        if key.startswith("biophysd_"):
            key = key[len("biophysd_") :]
        results[key] = first_last
        print(f"{key}: {first_last}")

    return results


def main() -> None:
    data_dir = Path("data/Excitatory_for_timestamps")
    output_path = Path("results") / "excitatory_time_stamp_edges"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    edges_dict = extract_time_stamp_edges(data_dir)
    np.save(output_path, edges_dict, allow_pickle=True)
    print(f"Saved {len(edges_dict)} entries to {output_path}")


if __name__ == "__main__":
    main()
