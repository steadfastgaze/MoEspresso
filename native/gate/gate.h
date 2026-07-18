// MoEspresso native gate: an MLX primitive that encodes an
// MTLSharedEvent wait on the live command stream. Kernels encoded after it
// execute only once the host signals the event with a value >= `value`.
//
// Product use: one gate per decode layer in front of the
// routed island; the IO worker signals after ensure+publish. Kernels wait
// for IO; threads never wait for kernels.
//
// The gate takes TWO inputs: the array it passes through AND an ordering
// token (the ring-export output). The token input forces the export kernel
// to be encoded BEFORE the gate; without it MLX's topological order could
// encode the gate first and deadlock (the worker would never see the
// indices it needs to load and signal).

#pragma once

#include <cstdint>

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mx = mlx::core;

namespace moespresso_gate {

// Pass x through; everything encoded after waits for signal >= value.
// `token` is consumed only for encode ordering (see header comment).
mx::array gate(
    const mx::array& x,
    const mx::array& token,
    uint64_t value,
    mx::StreamOrDevice s = {});

// Host-side signal / inspection (no MLX calls, callable from any thread).
void signal_event(uint64_t value);
uint64_t signaled_value();

class GateWait : public mx::Primitive {
 public:
  explicit GateWait(mx::Stream stream, uint64_t value)
      : mx::Primitive(stream), value_(value) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "MoEspressoGateWait";
  }

 private:
  uint64_t value_;
};

} // namespace moespresso_gate
