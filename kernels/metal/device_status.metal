#include <metal_stdlib>
using namespace metal;

kernel void check_finite_bf16(device const ushort *values [[buffer(0)]],
                              device atomic_uint *status [[buffer(1)]],
                              uint index [[thread_position_in_grid]]) {
    if ((values[index] & 0x7f80u) == 0x7f80u) {
        atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
    }
}

kernel void check_finite_f32(device const uint *values [[buffer(0)]],
                             device atomic_uint *status [[buffer(1)]],
                             uint index [[thread_position_in_grid]]) {
    if ((values[index] & 0x7f800000u) == 0x7f800000u) {
        atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
    }
}
