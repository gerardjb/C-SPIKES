"""Run delvewheel without reusing CUDA section padding for DLL names."""
from contextlib import contextmanager


@contextmanager
def preserve_cuda_sections():
    from delvewheel import _dll_utils

    # delvewheel 1.13.1/1.13.2 consider CUDA sections ordinary initialized
    # data. Appending DLL names there extends their virtual size and corrupts
    # the fatbin stream seen by cuobjdump. The same predicate drives both
    # padding allocation and writing; exclude .nv_fatb and .nvFatBi from both.
    # Normal DLL mangling remains enabled, using other padding or a new section.
    original = _dll_utils._are_characteristics_suitable

    def suitable(section):
        return not section.Name.startswith(b".nv") and original(section)

    _dll_utils._are_characteristics_suitable = suitable
    try:
        yield
    finally:
        _dll_utils._are_characteristics_suitable = original


if __name__ == "__main__":
    from delvewheel.__main__ import main

    with preserve_cuda_sections():
        main()
