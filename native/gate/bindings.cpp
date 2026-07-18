// MoEspresso native gate bindings.

#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>

#include "gate.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_moespresso_gate, m) {
  m.doc() = "MoEspresso MTLSharedEvent gate for MLX streams";

  m.def(
      "gate",
      &moespresso_gate::gate,
      "x"_a,
      "token"_a,
      "value"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      "Pass-through; kernels encoded after it wait for signal >= value. "
      "`token` forces encode-ordering after the ring export.");
  m.def("signal_event", &moespresso_gate::signal_event, "value"_a);
  m.def("signaled_value", &moespresso_gate::signaled_value);
}
