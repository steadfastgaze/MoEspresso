// Virtual Qwen GDN input projections over their original Q6_K matrices.
// Compiled with kquant.metal's unchanged block decoder and matvec fold.
kernel void qwen4_gdn_input_front_q6_k(
    device const uchar* qkv_weight [[buffer(0)]],
    device const uchar* gate_weight [[buffer(1)]],
    device const uchar* beta_weight [[buffer(2)]],
    device const uchar* decay_weight [[buffer(3)]],
    device const ushort* hidden [[buffer(4)]],
    device uchar* qkv [[buffer(5)]],
    device uchar* gate [[buffer(6)]],
    device uchar* beta [[buffer(7)]],
    device uchar* decay [[buffer(8)]],
    uint group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    if (group < 1280) {
        kq_q6_matvec<false>(qkv_weight, hidden, qkv, 2560, 10240, group, sg, lane);
    } else if (group < 2048) {
        kq_q6_matvec<false>(gate_weight, hidden, gate, 2560, 6144, group - 1280, sg, lane);
    } else if (group < 2054) {
        kq_q6_matvec<false>(beta_weight, hidden, beta, 2560, 48, group - 2048, sg, lane);
    } else if (group < 2060) {
        kq_q6_matvec<false>(decay_weight, hidden, decay, 2560, 48, group - 2054, sg, lane);
    }
}
