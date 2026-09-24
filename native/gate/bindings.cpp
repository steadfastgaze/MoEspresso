// MoEspresso native gate bindings.

#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>
#include "mlx/backend/metal/device.h"

#include "all_hit.h"
#include "gate.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_moespresso_gate, m) {
  m.doc() = "MoEspresso MTLSharedEvent gate for MLX streams";
  m.attr("QWEN_ALL_HIT_ABI") = moespresso_gate::all_hit::kAbi;
  m.def("submission_limits", []() {
    const auto [ops, mb] = mlx::core::metal::device(mlx::core::Device::gpu)
                               .get_max_ops_mb_per_buffer();
    nb::dict result;
    result["max_ops_per_buffer"] = ops;
    result["max_mb_per_buffer"] = mb;
    return result;
  }, "Read resolved MLX command-buffer thresholds without changing them.");

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
  m.def(
      "read_exported_ids", &moespresso_gate::all_hit::read_exported_ids,
      "ring"_a, "sequence"_a, "count"_a, "wait_ns"_a,
      "Read up to 64 checksummed route IDs without the GIL; return None while pending. "
      "The caller retains cancellation and timeout ownership; wait_ns is at most 1000000.");
  nb::class_<moespresso_gate::all_hit::AllHitAttempt>(m, "AllHitAttempt")
      .def(
          nb::init<
              nb::handle,
              nb::handle,
              nb::handle,
              nb::handle,
              nb::handle>(),
          "ring"_a,
          "maps"_a,
          "destinations"_a,
          "sequence"_a,
          "capacities"_a)
      .def("poll", &moespresso_gate::all_hit::AllHitAttempt::poll, "slice_ns"_a)
      .def("close", &moespresso_gate::all_hit::AllHitAttempt::close);
}
