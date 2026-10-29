# Owned brightness indices

`GCaMP::brightStates` is `csp_model::BrightIndices5`, an owned
`std::array<std::uint64_t,5>`. This changes the native field type and ABI: rebuild
native clients and replace Armadillo expressions/subviews. Python exports are
unchanged. Defaults remain {2,3,5,6,8}; ordering, duplicates and edits between
calls remain observable.

Host gathers and reductions preserve the prior arithmetic and access order using
owned values. There is no persistent cache. Direct reductions check/read each
coefficient before adding it; materialized gathers check each pair before either
read. The distinct orders matter for partial access and exception behavior.

No global physical-index restriction is added. Twelve-element getters and
nine-element operations retain their respective bounds behavior. Invalid indices
on unchecked scalar evolution loops remain outside supported usage. Concurrent
mutation of a shared model is unsupported. The separate particle-array kernel's
hard-coded brightness list is unchanged. Test-only Armadillo index bridges require
the recorded 64-bit index width; that is not a promise of cross-ABI compatibility.
