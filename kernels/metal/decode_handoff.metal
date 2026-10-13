#include <metal_stdlib>
using namespace metal;

kernel void export_decode_routes(
    const device uint* ids [[buffer(0)]],
    const device ushort* scores [[buffer(1)]],
    const device uint* changed [[buffer(2)]],
    const device atomic_uint* status [[buffer(3)]],
    device uint* mailbox [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    for (uint i = 0; i < 10; ++i) { mailbox[i] = ids[i]; mailbox[10 + i] = scores[i]; }
    mailbox[20] = changed[0];
    mailbox[21] = atomic_load_explicit(status, memory_order_relaxed);
}
