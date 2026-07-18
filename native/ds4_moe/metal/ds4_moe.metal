#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

template <typename RowT, typename ScoreT>
[[kernel]] void weighted_sum6(
    device const RowT* rows [[buffer(0)]],
    device const ScoreT* scores [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant const uint& width [[buffer(3)]],
    constant const uint& score_rows [[buffer(4)]],
    uint2 pos [[thread_position_in_grid]]) {
  uint col = pos.x;
  uint token = pos.y;
  uint score_token = score_rows == 1 ? 0 : token;
  float acc = 0.0f;
  for (uint k = 0; k < 6; k++) {
    uint row_offset = ((token * 6 + k) * width) + col;
    uint score_offset = score_token * 6 + k;
    acc += static_cast<float>(rows[row_offset]) *
        static_cast<float>(scores[score_offset]);
  }
  out[token * width + col] = acc;
}

#define instantiate_weighted_sum6(row_name, row_type, score_name, score_type) \
  instantiate_kernel(                                                         \
      "weighted_sum6_" #row_name "_" #score_name,                             \
      weighted_sum6,                                                          \
      row_type,                                                               \
      score_type)

#define instantiate_weighted_sum6_scores(row_name, row_type)                  \
  instantiate_weighted_sum6(row_name, row_type, float32, float)               \
  instantiate_weighted_sum6(row_name, row_type, float16, half)                \
  instantiate_weighted_sum6(row_name, row_type, bfloat16, bfloat16_t)

instantiate_weighted_sum6_scores(float32, float)
instantiate_weighted_sum6_scores(float16, half)
instantiate_weighted_sum6_scores(bfloat16, bfloat16_t)
