// BF16 residual composition around PLE and the final residual read.
#include <metal_stdlib>
using namespace metal;

kernel void prefill_residual_write(
    const device bfloat* residual [[buffer(0)]],
    const device bfloat* output [[buffer(1)]],
    const device bfloat* injection [[buffer(2)]],
    device bfloat* result [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    uint row = i / 10240u;
    uint branch = (i % 10240u) / 2560u;
    uint channel = i % 2560u;
    bfloat product = output[row * 2560u + channel] * injection[row * 4u + branch];
    result[i] = residual[i] + product;
}

kernel void prefill_add(
    const device bfloat* left [[buffer(0)]],
    const device bfloat* right [[buffer(1)]],
    device bfloat* result [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
    if (i < count) result[i] = left[i] + right[i];
}
