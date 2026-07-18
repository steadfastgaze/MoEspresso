// MoEspresso native gate implementation. See gate.h.

#include "gate.h"

#include <mutex>

#include "mlx/backend/metal/device.h"
#include "mlx/utils.h"

namespace moespresso_gate {

namespace {

MTL::SharedEvent* g_event = nullptr;
std::once_flag g_event_once;

MTL::SharedEvent* shared_event() {
  std::call_once(g_event_once, []() {
    auto& d = mx::metal::device(mx::Device::gpu);
    g_event = d.mtl_device()->newSharedEvent();
    g_event->setSignaledValue(0);
  });
  return g_event;
}

} // namespace

mx::array gate(
    const mx::array& x,
    const mx::array& token,
    uint64_t value,
    mx::StreamOrDevice s) {
  return mx::array(
      x.shape(),
      x.dtype(),
      std::make_shared<GateWait>(mx::to_stream(s), value),
      {x, token});
}

void signal_event(uint64_t value) {
  shared_event()->setSignaledValue(value);
}

uint64_t signaled_value() {
  return shared_event()->signaledValue();
}

void GateWait::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  // CPU fallback: no gating semantics, plain pass-through.
  outputs[0].copy_shared_buffer(inputs[0]);
}

void GateWait::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& enc = mx::metal::get_command_encoder(s);
  // End the live compute encoder (CommandEncoder resets and recreates on
  // next use), then encode the event wait on the underlying command buffer:
  // everything encoded after this point on the stream waits until the host
  // signals >= value_. Proven by the G0 spike: hold + 0.44 ms release +
  // multi-kernel integrity, no fence corruption.
  enc.end_encoding();
  enc.get_command_buffer()->encodeWait(shared_event(), value_);
  outputs[0].copy_shared_buffer(inputs[0]);
}

} // namespace moespresso_gate
