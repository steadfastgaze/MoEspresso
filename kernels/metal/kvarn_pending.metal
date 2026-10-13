// Overlay exact pending token-major rows on a committed KVarN gather.
#include <metal_stdlib>
using namespace metal;

// Validate a device selector before any ID participates in gather addressing.
// The sticky status is inspected at the completed session frontier.
kernel void kvarn_pending_selection_guard(
    const device int* logical_ids [[buffer(0)]],
    const device uchar* selected_valid [[buffer(1)]],
    const device uint* selected_count [[buffer(2)]],
    const constant uint* counters [[buffer(3)]],
    device uchar* effective_valid [[buffer(4)]],
    device uint* safe_count [[buffer(5)]],
    device atomic_uint* status [[buffer(6)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= 2051u) return;
    const uint count = selected_count[0];
    const bool count_ok = count == counters[2];
    if (row == 0u) safe_count[0] = count_ok ? count : 0u;
    const uint visible = counters[0] + counters[1];
    const bool active = row < min(count, 2051u);
    const int logical = logical_ids[row];
    const bool valid = active && selected_valid[row] == 1u
        && logical >= 0 && uint(logical) < visible;
    const bool padded = !active && selected_valid[row] == 0u && logical == -1;
    if (!count_ok || count == 0u || count > 2051u || !(valid || padded)) {
        atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
    }
    effective_valid[row] = valid ? 1u : 0u;
}

kernel void kvarn_pending_overlay(
    const device ushort* pending_keys [[buffer(0)]],
    const device ushort* pending_values [[buffer(1)]],
    const device int* logical_ids [[buffer(2)]],
    const device uchar* valid [[buffer(3)]],
    const constant uint* counters [[buffer(4)]],
    device ushort* keys [[buffer(5)]],
    device ushort* values [[buffer(6)]],
    uint i [[thread_position_in_grid]]) {
    constexpr uint width = 2051u;
    if (i >= 2u * width * 256u) return;
    const uint channel = i % 256u;
    const uint row = (i / 256u) % width;
    const uint head = i / (width * 256u);
    const int token = logical_ids[row] - int(counters[0]);
    if (valid[row] && token >= 0 && uint(token) < counters[1]) {
        const uint source = (uint(token) * 2u + head) * 256u + channel;
        keys[i] = pending_keys[source];
        values[i] = pending_values[source];
    }
}

// Tile forms retain the same guard and pending overlay semantics while one
// launch covers every query in the bounded prefill tile.
kernel void kvarn_pending_selection_guard_tile(
    const device int* logical_ids [[buffer(0)]],
    const device uchar* selected_valid [[buffer(1)]],
    const device uint* selected_count [[buffer(2)]],
    const device uint* query_counters [[buffer(3)]],
    device uchar* effective_valid [[buffer(4)]],
    device uint* safe_count [[buffer(5)]],
    device atomic_uint* status [[buffer(6)]],
    const constant uint* rows [[buffer(7)]],
    uint index [[thread_position_in_grid]]) {
    constexpr uint width = 2051u;
    const uint query = index / width, slot = index % width;
    if (query >= rows[0]) return;
    const device uint* counters = query_counters + query * 3u;
    const uint count = selected_count[query];
    const bool count_ok = count == counters[2];
    if (slot == 0u) safe_count[query] = count_ok ? count : 0u;
    const int logical = logical_ids[index];
    const bool active = slot < min(count, width);
    const bool valid = active && selected_valid[index] == 1u
        && logical >= 0 && uint(logical) < counters[0] + counters[1];
    const bool padded = !active && selected_valid[index] == 0u && logical == -1;
    if (!count_ok || count == 0u || count > width || !(valid || padded))
        atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
    effective_valid[index] = valid ? 1u : 0u;
}

kernel void kvarn_pending_overlay_tile(
    const device ushort* pending_keys [[buffer(0)]],
    const device ushort* pending_values [[buffer(1)]],
    const device int* logical_ids [[buffer(2)]],
    const device uchar* valid [[buffer(3)]],
    const device uint* query_counters [[buffer(4)]],
    device ushort* keys [[buffer(5)]],
    device ushort* values [[buffer(6)]],
    const constant uint* rows [[buffer(7)]],
    uint index [[thread_position_in_grid]]) {
    constexpr uint width = 2051u, row_words = 2u * width * 256u;
    const uint query = index / row_words, local = index % row_words;
    if (query >= rows[0]) return;
    const device uint* counters = query_counters + query * 3u;
    const uint channel = local % 256u;
    const uint row = (local / 256u) % width;
    const uint head = local / (width * 256u);
    const int token = logical_ids[query * width + row] - int(counters[0]);
    if (valid[query * width + row] && token >= 0 && uint(token) < counters[1]) {
        const uint source = (uint(token) * 2u + head) * 256u + channel;
        keys[index] = pending_keys[source];
        values[index] = pending_values[source];
    }
}
