// QSA device index compresses four rows into one BF16 row.
// Score, radix selection and expansion adapt the DS4 Qwen Metal path (MIT).
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

inline float qi_value(ushort bits) { return as_type<float>(uint(bits) << 16u); }
inline ushort qi_bf16(float value) {
    const uint bits = as_type<uint>(value);
    return ushort((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
}
inline void qi_bad(device atomic_uint* status) {
    atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
}

kernel void qsa_device_copy(
    const device uchar* source [[buffer(0)]],
    device uchar* target [[buffer(1)]],
    uint index [[thread_position_in_grid]]
) { target[index] = source[index]; }

kernel void qsa_device_pending(
    const device ushort* raw [[buffer(0)]],
    device ushort* pending [[buffer(1)]],
    const constant uint* slot [[buffer(2)]],
    device atomic_uint* status [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    const ushort value = raw[index];
    if ((value & 0x7f80u) == 0x7f80u || slot[0] >= 3u) qi_bad(status);
    else pending[slot[0] * 128u + index] = value;
}

kernel void qsa_device_compress(
    const device ushort* pending [[buffer(0)]],
    const device ushort* raw [[buffer(1)]],
    const device ushort* weight [[buffer(2)]],
    const device ushort* cosine [[buffer(3)]],
    const device ushort* sine [[buffer(4)]],
    device ushort* compressed [[buffer(5)]],
    const constant uint* group [[buffer(6)]],
    device atomic_uint* status [[buffer(7)]],
    uint tid [[thread_position_in_threadgroup]]
) {
    float mean[4];
    float partial = 0.0f;
    for (uint i = 0u; i < 4u; ++i) {
        const uint channel = tid * 4u + i;
        float total = 0.0f;
        for (uint row = 0u; row < 3u; ++row) {
            const ushort bits = pending[row * 128u + channel];
            if ((bits & 0x7f80u) == 0x7f80u) qi_bad(status);
            total += qi_value(bits);
        }
        if ((raw[channel] & 0x7f80u) == 0x7f80u) qi_bad(status);
        total += qi_value(raw[channel]);
        mean[i] = qi_value(qi_bf16(total * 0.25f));
        partial += mean[i] * mean[i];
    }
    const float inv = precise::rsqrt(simd_sum(partial) / 128.0f + 1e-6f);
    threadgroup ushort normalized[128];
    for (uint i = 0u; i < 4u; ++i) {
        const uint channel = tid * 4u + i;
        normalized[channel] = qi_bf16(mean[i] * inv * (1.0f + qi_value(weight[channel])));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0u; i < 4u; ++i) {
        const uint channel = tid * 4u + i;
        ushort result = normalized[channel];
        if (channel < 64u) {
            const uint peer = channel < 32u ? channel + 32u : channel - 32u;
            const float rotated = qi_value(normalized[peer]) * (channel < 32u ? -1.0f : 1.0f);
            const float first = qi_value(qi_bf16(qi_value(result) * qi_value(cosine[channel])));
            const float second = qi_value(qi_bf16(rotated * qi_value(sine[channel])));
            result = qi_bf16(first + second);
        }
        if ((result & 0x7f80u) == 0x7f80u) qi_bad(status);
        compressed[group[0] * 128u + channel] = result;
    }
}

// One SIMD group owns each newly completed four-token block. Only the first
// block may consume the old incomplete pending group; subsequent rows come
// directly from the projected chunk. RoPE uses that block's first position.
kernel void qsa_device_compress_chunk(
    const device ushort* pending [[buffer(0)]],
    const device ushort* raw [[buffer(1)]],
    const device ushort* weight [[buffer(2)]],
    device ushort* compressed [[buffer(3)]],
    device atomic_uint* status [[buffer(4)]],
    const constant uint* args [[buffer(5)]],
    uint block [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
    const uint frontier = args[0], rows = args[1], old_groups = args[2], new_groups = args[3];
    const uint absolute_group = old_groups + block;
    if (absolute_group >= new_groups) return;
    const uint first = absolute_group * 4u;
    float mean[4];
    float partial = 0.0f;
    for (uint i = 0u; i < 4u; ++i) {
        const uint channel = tid * 4u + i;
        float total = 0.0f;
        for (uint row = 0u; row < 4u; ++row) {
            const uint absolute = first + row;
            if (absolute >= frontier + rows) { qi_bad(status); continue; }
            const ushort bits = absolute < frontier
                ? pending[(absolute - old_groups * 4u) * 128u + channel]
                : raw[(absolute - frontier) * 128u + channel];
            if ((bits & 0x7f80u) == 0x7f80u) qi_bad(status);
            total += qi_value(bits);
        }
        mean[i] = qi_value(qi_bf16(total * 0.25f));
        partial += mean[i] * mean[i];
    }
    const float inv = precise::rsqrt(simd_sum(partial) / 128.0f + 1e-6f);
    threadgroup ushort normalized[128];
    for (uint i = 0u; i < 4u; ++i) {
        const uint channel = tid * 4u + i;
        normalized[channel] = qi_bf16(mean[i] * inv * (1.0f + qi_value(weight[channel])));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0u; i < 4u; ++i) {
        const uint channel = tid * 4u + i;
        ushort result = normalized[channel];
        if (channel < 64u) {
            const uint peer = channel < 32u ? channel + 32u : channel - 32u;
            const float frequency = float(channel % 32u) / 32.0f;
            const float inverse = metal::pow(10000000.0f, -frequency);
            const float angle = float(first) * inverse;
            const float cosine = qi_value(qi_bf16(metal::precise::cos(angle)));
            const float sine = qi_value(qi_bf16(metal::precise::sin(angle)));
            const float rotated = qi_value(normalized[peer]) * (channel < 32u ? -1.0f : 1.0f);
            const float a = qi_value(qi_bf16(qi_value(result) * cosine));
            const float b = qi_value(qi_bf16(rotated * sine));
            result = qi_bf16(a + b);
        }
        if ((result & 0x7f80u) == 0x7f80u) qi_bad(status);
        compressed[absolute_group * 128u + channel] = result;
    }
}

kernel void qsa_device_pending_tail(
    const device ushort* raw [[buffer(0)]],
    device ushort* pending [[buffer(1)]],
    device atomic_uint* status [[buffer(2)]],
    const constant uint* args [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    const uint frontier = args[0], next = args[1];
    const uint slot = index / 128u, channel = index % 128u;
    const uint absolute = next - next % 4u + slot;
    if (slot >= next % 4u || absolute < frontier) return;
    const ushort bits = raw[(absolute - frontier) * 128u + channel];
    if ((bits & 0x7f80u) == 0x7f80u) qi_bad(status);
    pending[index] = bits;
}

kernel void qsa_device_score(
    const device ushort* query [[buffer(0)]],
    const device ushort* compressed [[buffer(1)]],
    device float* scores [[buffer(2)]],
    const constant uint* groups [[buffer(3)]],
    device atomic_uint* status [[buffer(4)]],
    uint block [[thread_position_in_grid]]
) {
    if (block >= groups[0]) return;
    const device ushort* key = compressed + block * 128u;
    float score = 0.0f;
    for (uint head = 0u; head < 4u; ++head) {
        float dot = 0.0f;
        for (uint channel = 0u; channel < 128u; ++channel) {
            dot += qi_value(query[head * 128u + channel]) * qi_value(key[channel]);
        }
        if (!metal::isfinite(dot)) {
            qi_bad(status);
            dot = 0.0f;
        }
        score += metal::max(dot, 0.0f);
    }
    score *= 0.08838834764831845f;
    if (!metal::isfinite(score) || score < 0.0f) {
        qi_bad(status);
        score = 0.0f;
    }
    scores[block] = score;
}

// Adapted from DS4's 16-query by 64-group index matrix scorer (MIT).
// The cache and projected queries are BF16; matrix operands are
// staged as half while the accumulator and per-head ReLU remain FP32.
kernel void qsa_device_score_mm(
    const device ushort* query [[buffer(0)]],
    const device ushort* compressed [[buffer(1)]],
    device float* scores [[buffer(2)]],
    device atomic_uint* status [[buffer(3)]],
    const constant uint* args [[buffer(4)]],
    uint group_index [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort simd_group [[simdgroup_index_in_threadgroup]]
) {
    const uint rows = args[0], groups = args[1], nblock_tiles = (groups + 63u) / 64u;
    const uint b0 = (group_index % nblock_tiles) * 64u;
    const uint t0 = (group_index / nblock_tiles) * 16u;
    if (b0 >= groups || t0 >= rows) return;
    threadgroup uint4 storage[1040];
    threadgroup half* keys = (threadgroup half*)storage;
    threadgroup half* transposed_queries = keys + 64u * 64u;
    threadgroup float* product = (threadgroup float*)storage;
    const uint block_offset = (simd_group >> 1u) * 32u;
    const uint row_offset = (simd_group & 1u) * 32u;
    simdgroup_float8x8 accumulators[4][4];
    for (uint i = 0u; i < 4u; ++i)
        for (uint j = 0u; j < 4u; ++j)
            accumulators[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k0 = 0u; k0 < 128u; k0 += 64u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < 64u * 64u; i += 128u) {
            const uint block = b0 + i / 64u;
            const uint channel = k0 + i % 64u;
            const ushort bits = block < groups ? compressed[block * 128u + channel] : ushort(0);
            if ((bits & 0x7f80u) == 0x7f80u) qi_bad(status);
            keys[i] = (half)qi_value(bits);
        }
        for (uint i = tid; i < 64u * 64u; i += 128u) {
            const uint channel = i / 64u;
            const uint lane = i % 64u;
            const uint token = t0 + lane / 4u;
            const uint head = lane % 4u;
            const ushort bits = token < rows ? query[token * 512u + head * 128u + k0 + channel] : ushort(0);
            if ((bits & 0x7f80u) == 0x7f80u) qi_bad(status);
            transposed_queries[channel * 66u + lane] = (half)qi_value(bits);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0u; kk < 64u; kk += 8u) {
            simdgroup_half8x8 a[4], b[4];
            for (uint i = 0u; i < 4u; ++i)
                simdgroup_load(a[i], keys + (block_offset + i * 8u) * 64u + kk, 64, 0, false);
            for (uint j = 0u; j < 4u; ++j)
                simdgroup_load(b[j], transposed_queries + kk * 66u + row_offset + j * 8u, 66, 0, false);
            for (uint i = 0u; i < 4u; ++i)
                for (uint j = 0u; j < 4u; ++j)
                    simdgroup_multiply_accumulate(accumulators[i][j], a[i], b[j], accumulators[i][j]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0u; i < 4u; ++i)
        for (uint j = 0u; j < 4u; ++j)
            simdgroup_store(accumulators[i][j],
                product + (row_offset + j * 8u) * 64u + block_offset + i * 8u, 64, 0, true);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < 16u * 64u; i += 128u) {
        const uint token = t0 + i / 64u, block = b0 + i % 64u;
        if (token >= rows || block >= groups) continue;
        const uint visible = (args[2] + token + 1u) / 4u;
        float score = 0.0f;
        if (block < visible) {
            for (uint head = 0u; head < 4u; ++head) {
                const float dot = product[((token - t0) * 4u + head) * 64u + block - b0];
                if (!metal::isfinite(dot)) qi_bad(status);
                score += metal::max(dot, 0.0f);
            }
            score *= 0.08838834764831845f;
            if (!metal::isfinite(score)) qi_bad(status);
        }
        scores[token * groups + block] = score;
    }
}

// Adapted from DS4's ordinary 1024-thread four-pass radix selector.
kernel void qsa_device_select(
    const device float* scores [[buffer(0)]],
    const constant uint* groups [[buffer(1)]],
    device int* selected [[buffer(2)]],
    device atomic_uint* status [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    const uint n = groups[0], nth = 1024u, top_k = 512u;
    if (n <= top_k || n > 65536u) { if (tid == 0u) qi_bad(status); return; }
    threadgroup atomic_uint hist[256];
    threadgroup uint scan[32];
    threadgroup uint found[2];
    uint prefix = 0u, need = top_k;
    for (uint pass = 0u; pass < 4u; ++pass) {
        const uint shift = 24u - 8u * pass;
        const uint mask_hi = pass == 0u ? 0u : (0xffffffffu << (shift + 8u));
        if (tid < 256u) atomic_store_explicit(&hist[tid], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint block = tid; block < n; block += 16u * nth) {
            float values[16];
            for (uint i = 0u; i < 16u; ++i) {
                const uint at = block + i * nth;
                values[i] = at < n ? scores[at] : 0.0f;
            }
            for (uint i = 0u; i < 16u; ++i) {
                const uint at = block + i * nth;
                if (at >= n) break;
                const uint key = as_type<uint>(metal::max(values[i], 0.0f));
                if ((key & mask_hi) == prefix)
                    atomic_fetch_add_explicit(&hist[(key >> shift) & 255u], 1u, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint count = tid < 256u ? atomic_load_explicit(&hist[255u - tid], memory_order_relaxed) : 0u;
        const uint local = simd_prefix_inclusive_sum(count);
        if (lane == 31u) scan[simd_group] = local;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 256u) {
            uint above = local - count;
            for (uint group = 0u; group < simd_group; ++group) above += scan[group];
            if (above < need && above + count >= need) {
                found[0] = prefix | ((255u - tid) << shift);
                found[1] = need - above;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prefix = found[0]; need = found[1];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint chunk = (n + nth - 1u) / nth;
    const uint begin = metal::min(tid * chunk, n), end = metal::min(begin + chunk, n);
    uint greater = 0u, equal = 0u;
    for (uint block = begin; block < end; ++block) {
        const uint key = as_type<uint>(metal::max(scores[block], 0.0f));
        greater += uint(key > prefix);
        equal += uint(key == prefix);
    }
    uint rank_greater = 0u, rank_equal = 0u;
    for (uint which = 0u; which < 2u; ++which) {
        const uint count = which == 0u ? greater : equal;
        const uint local = simd_prefix_exclusive_sum(count);
        if (lane == 31u) scan[simd_group] = local + count;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_group == 0u) {
            const uint prior = simd_prefix_exclusive_sum(scan[lane]);
            scan[lane] = prior;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (which == 0u) rank_greater = local + scan[simd_group];
        else rank_equal = local + scan[simd_group];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint equals_base = top_k - need;
    for (uint block = begin; block < end; ++block) {
        const uint key = as_type<uint>(metal::max(scores[block], 0.0f));
        if (key > prefix) selected[rank_greater++] = int(block);
        else if (key == prefix) {
            if (rank_equal < need) selected[equals_base + rank_equal] = int(block);
            ++rank_equal;
        }
    }
}

// Multiquery DS4 radix grid: one independent 1024-thread group per row.
kernel void qsa_device_select_tile(
    const device float* scores [[buffer(0)]],
    const device uint* contexts [[buffer(1)]],
    device int* selected [[buffer(2)]],
    device atomic_uint* status [[buffer(3)]],
    const constant uint* args [[buffer(4)]],
    uint query [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    if (query >= args[1]) return;
    scores += query * args[0];
    selected += query * 512u;
    const uint n = contexts[query * 2u + 1u], nth = 1024u, top_k = 512u;
    if (n <= top_k) return;
    if (n > 65536u) { if (tid == 0u) qi_bad(status); return; }
    threadgroup atomic_uint hist[256];
    threadgroup uint scan[32];
    threadgroup uint found[2];
    uint prefix = 0u, need = top_k;
    for (uint pass = 0u; pass < 4u; ++pass) {
        const uint shift = 24u - 8u * pass;
        const uint mask_hi = pass == 0u ? 0u : (0xffffffffu << (shift + 8u));
        if (tid < 256u) atomic_store_explicit(&hist[tid], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint block = tid; block < n; block += 16u * nth) {
            float values[16];
            for (uint i = 0u; i < 16u; ++i) {
                const uint at = block + i * nth;
                values[i] = at < n ? scores[at] : 0.0f;
            }
            for (uint i = 0u; i < 16u; ++i) {
                const uint at = block + i * nth;
                if (at >= n) break;
                const uint key = as_type<uint>(metal::max(values[i], 0.0f));
                if ((key & mask_hi) == prefix)
                    atomic_fetch_add_explicit(&hist[(key >> shift) & 255u], 1u, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint count = tid < 256u ? atomic_load_explicit(&hist[255u - tid], memory_order_relaxed) : 0u;
        const uint local = simd_prefix_inclusive_sum(count);
        if (lane == 31u) scan[simd_group] = local;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 256u) {
            uint above = local - count;
            for (uint group = 0u; group < simd_group; ++group) above += scan[group];
            if (above < need && above + count >= need) {
                found[0] = prefix | ((255u - tid) << shift);
                found[1] = need - above;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prefix = found[0]; need = found[1];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint chunk = (n + nth - 1u) / nth;
    const uint begin = metal::min(tid * chunk, n), end = metal::min(begin + chunk, n);
    uint greater = 0u, equal = 0u;
    for (uint block = begin; block < end; ++block) {
        const uint key = as_type<uint>(metal::max(scores[block], 0.0f));
        greater += uint(key > prefix);
        equal += uint(key == prefix);
    }
    uint rank_greater = 0u, rank_equal = 0u;
    for (uint which = 0u; which < 2u; ++which) {
        const uint count = which == 0u ? greater : equal;
        const uint local = simd_prefix_exclusive_sum(count);
        if (lane == 31u) scan[simd_group] = local + count;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_group == 0u) {
            const uint prior = simd_prefix_exclusive_sum(scan[lane]);
            scan[lane] = prior;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (which == 0u) rank_greater = local + scan[simd_group];
        else rank_equal = local + scan[simd_group];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint equals_base = top_k - need;
    for (uint block = begin; block < end; ++block) {
        const uint key = as_type<uint>(metal::max(scores[block], 0.0f));
        if (key > prefix) selected[rank_greater++] = int(block);
        else if (key == prefix) {
            if (rank_equal < need) selected[equals_base + rank_equal] = int(block);
            ++rank_equal;
        }
    }
}

kernel void qsa_device_expand(
    const device int* selected_groups [[buffer(0)]],
    device int* ids [[buffer(1)]],
    device uchar* valid [[buffer(2)]],
    device uint* count [[buffer(3)]],
    const constant uint* context [[buffer(4)]],
    device atomic_uint* status [[buffer(5)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= 2051u) return;
    const uint frontier = context[0], groups = context[1];
    const bool sparse = groups > 512u;
    const uint n = sparse ? 2048u + frontier % 4u : frontier;
    if (index == 0u) count[0] = n;
    int logical = -1;
    if (index < n) {
        if (!sparse) logical = int(index);
        else if (index < 2048u) {
            const int block = selected_groups[index / 4u];
            if (block < 0 || uint(block) >= groups) qi_bad(status);
            else logical = block * 4 + int(index % 4u);
        } else logical = int(groups * 4u + index - 2048u);
        if (logical < 0 || uint(logical) >= frontier) qi_bad(status);
    }
    ids[index] = logical;
    valid[index] = uchar(index < n && logical >= 0 && uint(logical) < frontier);
}

// One 256-thread group expands one query's 2051 selected-token slots.
kernel void qsa_device_expand_tile(
    const device int* selected_groups [[buffer(0)]],
    device int* ids [[buffer(1)]],
    device uchar* valid [[buffer(2)]],
    device uint* count [[buffer(3)]],
    const device uint* contexts [[buffer(4)]],
    device atomic_uint* status [[buffer(5)]],
    const constant uint* rows [[buffer(6)]],
    uint query [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
    if (query >= rows[0]) return;
    const uint frontier = contexts[query * 2u], groups = contexts[query * 2u + 1u];
    const bool sparse = groups > 512u;
    const uint n = sparse ? 2048u + frontier % 4u : frontier;
    if (tid == 0u) count[query] = n;
    for (uint index = tid; index < 2051u; index += 256u) {
        int logical = -1;
        if (index < n) {
            if (!sparse) logical = int(index);
            else if (index < 2048u) {
                const int block = selected_groups[query * 512u + index / 4u];
                if (block < 0 || uint(block) >= groups) qi_bad(status);
                else logical = block * 4 + int(index % 4u);
            } else logical = int(groups * 4u + index - 2048u);
            if (logical < 0 || uint(logical) >= frontier) qi_bad(status);
        }
        ids[query * 2051u + index] = logical;
        valid[query * 2051u + index] = uchar(index < n && logical >= 0 && uint(logical) < frontier);
    }
}
