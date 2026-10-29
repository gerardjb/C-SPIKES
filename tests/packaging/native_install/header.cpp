#include "application_arithmetic_compat.h"
#include <type_traits>
static_assert(sizeof(std::uint64_t)==8 && sizeof(std::int64_t)==8);
