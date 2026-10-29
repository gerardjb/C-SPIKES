# Armadillo-derived compatibility logic

The Gaussian expression evaluation order and OpenMP dispatch in model_evolution_compat.cpp adapt Armadillo eop_core_meat.hpp and mp_misc.hpp from the pinned dependency.

Copyright 2008-2016 Conrad Sanderson (http://conradsanderson.id.au)
Copyright 2008-2016 National ICT Australia (NICTA)

Licensed under the Apache License, Version 2.0. See the retained full license in tests/native/g2ah/LICENSE.armadillo and http://www.apache.org/licenses/LICENSE-2.0. The original source is provided without warranties or conditions, as described by that license. The local modifications replace expression-template containers with owned storage and reproduce the pinned ordering and dispatch; they are not upstream Armadillo code.
