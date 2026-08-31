from pathlib import Path

from c_spikes.gui.paths import configure_project_root
from c_spikes.tensorflow_env import preload_tensorflow_quietly

configure_project_root(Path(__file__).resolve().parents[1])

preload_tensorflow_quietly()

from c_spikes.gui.app import main


if __name__ == "__main__":
    main()
