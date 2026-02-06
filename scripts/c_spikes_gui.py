from c_spikes.tensorflow_env import preload_tensorflow_quietly

preload_tensorflow_quietly()

from c_spikes.gui.app import main


if __name__ == "__main__":
    main()
