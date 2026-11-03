"""Exercise DLL renaming on PE files with CUDA sections and limited padding."""
from pathlib import Path
import struct
import sys

import pytest

pytest.importorskip("delvewheel")
from delvewheel import _dll_utils
import pefile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from repair_windows_wheel import preserve_cuda_sections


def minimal_pe(cuda_section, data_padding):
    """A non-executable PE64 fixture importing gsl.dll by ordinal."""
    data = bytearray(1536)  # 512-byte headers and two 512-byte sections
    data[:2] = b"MZ"
    struct.pack_into("<I", data, 0x3C, 0x80)
    data[0x80:0x84] = b"PE\0\0"
    struct.pack_into("<HHIIIHH", data, 0x84, 0x8664, 2, 0, 0, 0, 240, 0x2022)
    optional = 0x98
    struct.pack_into("<H", data, optional, 0x20B)  # PE32+
    struct.pack_into("<I", data, optional + 8, 1024)  # initialized data
    struct.pack_into("<Q", data, optional + 24, 0x180000000)  # image base
    struct.pack_into("<II", data, optional + 32, 4096, 512)  # alignments
    struct.pack_into("<H", data, optional + 40, 6)  # OS version
    struct.pack_into("<II", data, optional + 56, 0x3000, 512)  # image, headers
    struct.pack_into("<H", data, optional + 68, 3)  # console subsystem
    struct.pack_into("<QQQQ", data, optional + 72, 1048576, 4096, 1048576, 4096)
    struct.pack_into("<I", data, optional + 108, 16)  # data directory count
    struct.pack_into("<II", data, optional + 120, 0x1000, 40)  # import table
    struct.pack_into("<II", data, optional + 208, 0x1090, 16)  # import addresses
    for index, (name, size) in enumerate(((b".rdata", 512 - data_padding), (cuda_section, 16))):
        struct.pack_into("<8sIIIIIIHHI", data, optional + 240 + 40 * index,
                         name, size, 0x1000 * (index + 1), 512, 512 * (index + 1),
                         0, 0, 0, 0, 0x40000040)  # readable initialized data
    struct.pack_into("<IIIII", data, 512, 0x1080, 0, 0, 0x1040, 0x1090)
    data[576:584] = b"gsl.dll\0"
    for offset in (640, 656):
        struct.pack_into("<QQ", data, offset, 0x8000000000000001, 0)
    struct.pack_into("<IHHQ", data, 1024, 0xBA55ED50, 1, 16, 0)
    return bytes(data)


@pytest.mark.parametrize("cuda_section", [b".nv_fatb", b".nvFatBi"])
@pytest.mark.parametrize("data_padding", [0, 128])
def test_dll_mangling_preserves_cuda_section(tmp_path, cuda_section, data_padding):
    path = tmp_path / "gpu.pyd"
    raw = minimal_pe(cuda_section, data_padding)
    path.write_bytes(raw)
    mangled = "gsl-" + "a" * 32 + ".dll"
    original = _dll_utils._are_characteristics_suitable
    with preserve_cuda_sections():
        _dll_utils.replace_needed(str(path), ["gsl.dll"], {"gsl.dll": mangled}, False, True)
    assert _dll_utils._are_characteristics_suitable is original
    with pefile.PE(str(path), fast_load=False) as pe:
        assert [entry.dll for entry in pe.DIRECTORY_ENTRY_IMPORT] == [mangled.encode()]
        section = next(s for s in pe.sections if s.Name.rstrip(b"\0") == cuda_section)
        assert section.Misc_VirtualSize == 16
        assert section.get_data() == raw[1024:1536]
        # Exercise both ordinary padding reuse and adding a separate section.
        assert pe.FILE_HEADER.NumberOfSections == (2 if data_padding else 3)
