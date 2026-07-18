// MoEspresso native DS4 routed-MoE bindings.

#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>

#include "ds4_moe.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_moespresso_ds4_moe, m) {
  m.doc() = "MoEspresso native DS4 routed-MoE kernels for MLX";

  m.def(
      "weighted_sum6",
      &moespresso_ds4_moe::weighted_sum6,
      "rows"_a,
      "scores"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      "Return sum over six selected expert rows with route scores.");
}
