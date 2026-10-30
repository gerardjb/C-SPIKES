Runtime libraries bundled by auditwheel/delvewheel may include libgfortran,
libgomp, libgcc_s and libquadmath. The accompanying GCC license texts are from
https://gcc.gnu.org/git/?p=gcc.git;a=tree;hb=releases/gcc-12.2.0
(COPYING3, COPYING.RUNTIME and COPYING.LIB).

libgfortran/libgomp/libgcc use GPLv3 with the GCC Runtime Library Exception;
libquadmath uses LGPLv2.1. The Windows wheel also includes the license directory
of the actual MinGW runtime supplied by vcpkg. Native dependency license notices
are in adjacent directories. These notices do not change the C-SPIKES license.
