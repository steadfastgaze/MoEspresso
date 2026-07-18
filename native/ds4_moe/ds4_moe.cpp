// Native DS4 routed-MoE MLX primitives.

#include "ds4_moe.h"

#include <dlfcn.h>

#include <filesystem>
#include <stdexcept>
#include <string>

#include "mlx/backend/cpu/encoder.h"
#include "mlx/utils.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#endif

namespace moespresso_ds4_moe {

namespace {

constexpr int kTopK = 6;

std::string current_binary_dir() {
  static std::string binary_dir = []() {
    Dl_info info;
    if (!dladdr(reinterpret_cast<void*>(&current_binary_dir), &info)) {
      throw std::runtime_error("Unable to get current binary dir.");
    }
    return std::filesystem::path(info.dli_fname).parent_path().string();
  }();
  return binary_dir;
}

void validate_inputs(const mx::array& rows, const mx::array& scores) {
  if (rows.ndim() != 3) {
    throw std::invalid_argument("rows must have shape [tokens, 6, hidden]");
  }
  if (rows.shape(1) != kTopK) {
    throw std::invalid_argument("rows top_k dimension must be 6");
  }
  if (scores.ndim() != 2) {
    throw std::invalid_argument("scores must have shape [1, 6] or [tokens, 6]");
  }
  if (scores.shape(1) != kTopK) {
    throw std::invalid_argument("scores top_k dimension must be 6");
  }
  if (scores.shape(0) != 1 && scores.shape(0) != rows.shape(0)) {
    throw std::invalid_argument(
        "scores must have one row or one row per token");
  }
  if (!mx::issubdtype(rows.dtype(), mx::floating)) {
    throw std::invalid_argument("rows must be floating point");
  }
  if (!mx::issubdtype(scores.dtype(), mx::floating)) {
    throw std::invalid_argument("scores must be floating point");
  }
}

} // namespace

mx::array weighted_sum6(
    const mx::array& rows,
    const mx::array& scores,
    mx::StreamOrDevice s) {
  validate_inputs(rows, scores);
  mx::Shape out_shape{rows.shape(0), rows.shape(2)};
  int score_rows = scores.shape(0) == 1 ? 1 : rows.shape(0);
  return mx::array(
      out_shape,
      mx::float32,
      std::make_shared<WeightedSum6>(mx::to_stream(s), score_rows),
      {rows, scores});
}

void WeightedSum6::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  throw std::runtime_error("MoEspressoDS4WeightedSum6 has no CPU backend.");
}

#ifdef _METAL_

void WeightedSum6::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  const auto& rows = inputs[0];
  const auto& scores = inputs[1];
  auto& out = outputs[0];
  auto& s = stream();
  auto& d = mx::metal::device(s.device);

  out.set_data(mx::allocator::malloc(out.nbytes()));

  std::string kname = "weighted_sum6_";
  kname += mx::type_to_name(rows);
  kname += "_";
  kname += mx::type_to_name(scores);

  auto lib = d.get_library("moespresso_ds4_moe", current_binary_dir());
  auto kernel = d.get_kernel(kname, lib);
  auto& encoder = mx::metal::get_command_encoder(s);
  encoder.set_compute_pipeline_state(kernel);

  uint32_t width = static_cast<uint32_t>(rows.shape(2));
  uint32_t score_rows = static_cast<uint32_t>(score_rows_);

  encoder.set_input_array(rows, 0);
  encoder.set_input_array(scores, 1);
  encoder.set_output_array(out, 2);
  encoder.set_bytes(width, 3);
  encoder.set_bytes(score_rows, 4);

  size_t cols = static_cast<size_t>(rows.shape(2));
  size_t tokens = static_cast<size_t>(rows.shape(0));
  size_t group_x = std::min(cols, kernel->maxTotalThreadsPerThreadgroup());
  MTL::Size group_dims = MTL::Size(group_x, 1, 1);
  MTL::Size grid_dims = MTL::Size(cols, tokens, 1);
  encoder.dispatch_threads(grid_dims, group_dims);
}

#else

void WeightedSum6::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  throw std::runtime_error("MoEspressoDS4WeightedSum6 has no GPU backend.");
}

#endif

} // namespace moespresso_ds4_moe
