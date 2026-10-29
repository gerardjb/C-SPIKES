# Third-party installation notices

`notices/` contains license/attribution text and a SHA-256 manifest for one
recorded native-core dependency set. These are legal distribution inputs, not
test-output fixtures. Preserve them while distributing the corresponding material.

For `CSP_NATIVE_CORE_NOTICES_DIR`, the packager must review the actual dependency
versions and verify the manifest before selecting this directory. CMake checks
that a manifest exists and copies the supplied directory; it does not establish
license completeness or verify the hashes. See
[the installation guide](../../../docs/native-core-installation.md).

There is no runnable installation/relocation test controller in this directory.
Default pytest does not check installed-prefix origins, relocation or numerical
symbol collisions. Those checks require an explicitly prepared validation profile.
