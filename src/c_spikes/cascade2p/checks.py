

"""
General checks to simplify jupyter notebooks and GUI
"""

from c_spikes.tensorflow_env import configure_tensorflow_environment


def check_packages():
    """ Wrapper for check_yaml and check_keras_version """
    check_yaml()
    check_keras_version()


def check_yaml():
    """ Check if ruamel.yaml is installed, otherwise notify user with instructions """

    try:
        import ruamel.yaml
    except ModuleNotFoundError:
        print('\nModuleNotFoundError: The package "ruamel.yaml" does not seem to be installed on this PC.',
              'This package is necessary to load the configuration files of the models.\n',
              'Please install it with "pip install ruamel.yaml"')
        return

    print('\tYAML reader installed (version {}).'.format(ruamel.yaml.__version__))

def check_keras_version():
    """ Import keras and tensorflow and check versions """
    configure_tensorflow_environment()
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ModuleNotFoundError:
        print('ModuleNotFoundError: The package "tensorflow" does not seem to be installed on this PC.',
              'Please install tensorflow with "pip install tensorflow==2.1.0".')
        return

    print('\tKeras installed (version {}).'.format(keras.__version__) )
    print('\tTensorflow installed (version {}).'.format(tf.__version__) )

    ## TODO: perform check that versions are compatible, notify user
