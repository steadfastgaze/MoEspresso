// Qwen shared gate, up and scalar gate projections over their original Q6_K rows.
// This source is compiled with kquant.metal's block decoder and matvec folds.
kernel void qwen4_shared_front_q6_k(
    device const uchar* gate_weight [[buffer(0)]],
    device const uchar* up_weight [[buffer(1)]],
    device const uchar* scalar_weight [[buffer(2)]],
    device const ushort* hidden [[buffer(3)]],
    device uchar* gate [[buffer(4)]],
    device uchar* up [[buffer(5)]],
    device uchar* scalar [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    if (group < 80) {
        kq_q6_matvec<false>(gate_weight, hidden, gate, 2560, 640, group, sg, lane);
    } else if (group < 160) {
        kq_q6_matvec<false>(up_weight, hidden, up, 2560, 640, group - 80, sg, lane);
    } else if (group == 160) {
        kq_q6_matvec<false>(scalar_weight, hidden, scalar, 2560, 1, 0, sg, lane);
    }
}
