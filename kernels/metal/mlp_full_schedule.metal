// Stable device grouping for the existing full-resident tile32 projections.
// Tile schedule fields retain the antirez/ds4 layout (MIT).
// See THIRD-PARTY-NOTICES.
#include <metal_stdlib>
using namespace metal;

// Each block reads its routes once. Histogram cells are overwritten on reuse.
kernel void mlp_full_histogram(
    device const uint* ids [[buffer(0)]], device uint* histogram [[buffer(1)]],
    constant uint& count [[buffer(2)]], uint block [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    threadgroup atomic_uint counts[512];
    for (uint e = lane; e < 512u; e += 256u)
        atomic_store_explicit(counts + e, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint pair = block * 256u + lane;
    if (pair < count) {
        const uint source = ids[pair];
        if (source < 512u)
            atomic_fetch_add_explicit(counts + source, 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = lane; e < 512u; e += 256u)
        histogram[block * 512u + e] = atomic_load_explicit(counts + e, memory_order_relaxed);
}

// One thread owns each source's ranges and source-ascending tile32 schedule.
kernel void mlp_full_offsets(
    device const uint* histogram [[buffer(0)]], device uint* block_starts [[buffer(1)]],
    device uint* schedule [[buffer(2)]], device uint* total [[buffer(3)]],
    constant uint& count [[buffer(4)]], uint source [[thread_index_in_threadgroup]]) {
    threadgroup uint counts[512], starts[512], offsets[512];
    const uint blocks = (count + 255u) / 256u;
    uint n = 0u;
    for (uint block = 0u; block < blocks; ++block) n += histogram[block * 512u + source];
    counts[source] = n;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (source == 0u) {
        uint start = 0u, tiles = 0u;
        for (uint e = 0u; e < 512u; ++e) {
            starts[e] = start;
            offsets[e] = tiles;
            start += counts[e];
            tiles += (counts[e] + 31u) / 32u;
        }
        total[0] = tiles;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint at = starts[source];
    for (uint block = 0u; block < blocks; ++block) {
        block_starts[block * 512u + source] = at;
        at += histogram[block * 512u + source];
    }
    for (uint tile = 0u; tile < (n + 31u) / 32u; ++tile) {
        const uint slot = offsets[source] + tile;
        schedule[slot * 3u] = source;
        schedule[slot * 3u + 1u] = starts[source] + tile * 32u;
        schedule[slot * 3u + 2u] = min(32u, n - tile * 32u);
    }
}

// Block prefixes plus earlier matching lanes retain ascending original pairs.
kernel void mlp_full_stable_order(
    device const uint* ids [[buffer(0)]], device const uint* block_starts [[buffer(1)]],
    device uint* order [[buffer(2)]], constant uint& count [[buffer(3)]],
    uint block [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    threadgroup uint sources[256];
    const uint pair = block * 256u + lane;
    const uint source = pair < count ? ids[pair] : 512u;
    sources[lane] = source;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (pair >= count || source >= 512u) return;
    uint rank = 0u;
    for (uint prior = 0u; prior < lane; ++prior) rank += sources[prior] == source;
    order[block_starts[block * 512u + source] + rank] = pair;
}
