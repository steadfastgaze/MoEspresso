// Native DS4 routed-MoE MLX primitives.

#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mx = mlx::core;

namespace moespresso_ds4_moe {

mx::array weighted_sum6(
    const mx::array& rows,
    const mx::array& scores,
    mx::StreamOrDevice s = {});

class WeightedSum6 : public mx::Primitive {
 public:
  explicit WeightedSum6(mx::Stream stream, int score_rows)
      : mx::Primitive(stream), score_rows_(score_rows) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "MoEspressoDS4WeightedSum6";
  }

 private:
  int score_rows_;
};

} // namespace moespresso_ds4_moe
