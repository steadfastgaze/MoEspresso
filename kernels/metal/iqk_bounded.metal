
// Publication-gated routed experts use the numerical helpers above.
// A failed gate exits uniformly before selectors, expert rows, or barriers.
#define BOUNDED_GATE(NAME, IQ3) \
kernel void NAME( \
    const device half* x [[buffer(0)]], \
    const device uchar* gqs [[buffer(1)]], const device uint* gqh [[buffer(2)]], \
    const device uchar* gscl [[buffer(3)]], const device uchar* gsch [[buffer(4)]], \
    const device uchar* gsex [[buffer(5)]], const device half* gdv [[buffer(6)]], \
    const device half* table [[buffer(7)]], \
    const device uchar* uqs [[buffer(8)]], const device uint* uqh [[buffer(9)]], \
    const device uchar* uscl [[buffer(10)]], const device uchar* usch [[buffer(11)]], \
    const device uchar* usex [[buffer(12)]], const device half* udv [[buffer(13)]], \
    const device uint* slots [[buffer(14)]], const device uint* order [[buffer(15)]], \
    const device half* sigmoid [[buffer(16)]], device half* gate [[buffer(17)]], \
    device half* up [[buffer(18)]], device half* activation [[buffer(19)]], \
    const device uint* publication [[buffer(20)]], \
    uint tid [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]]) { \
    if (publication[0] != 1u) return; \
    threadgroup half values[IQ3 ? 16 : 8]; \
    threadgroup float gp[48], upart[48]; \
    routed_gate_up<IQ3>(x, gqs, gqh, gscl, gsch, gsex, gdv, table, \
        uqs, uqh, uscl, usch, usex, udv, slots, order, sigmoid, gate, up, activation, \
        values, gp, upart, tid, group); \
}
#define BOUNDED_DOWN(NAME, IQ3) \
kernel void NAME( \
    const device half* x [[buffer(0)]], \
    const device uchar* qs [[buffer(1)]], const device uint* qh [[buffer(2)]], \
    const device uchar* scl [[buffer(3)]], const device uchar* sch [[buffer(4)]], \
    const device uchar* sex [[buffer(5)]], const device half* dv [[buffer(6)]], \
    const device half* table [[buffer(7)]], const device uint* slots [[buffer(8)]], \
    const device ushort* scores [[buffer(9)]], const device uint* order [[buffer(10)]], \
    device half* down [[buffer(11)]], device ushort* output [[buffer(12)]], \
    const device uint* publication [[buffer(13)]], \
    uint tid [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]]) { \
    if (publication[0] != 1u) { if (tid < 16u) output[group * 16u + tid] = 0u; return; } \
    threadgroup half values[IQ3 ? 16 : 8], slot_values[160]; \
    routed_down<IQ3>(x, qs, qh, scl, sch, sex, dv, table, slots, scores, order, \
        down, output, values, slot_values, tid, group); \
}
BOUNDED_GATE(bounded_gate_iq2, false)
BOUNDED_GATE(bounded_gate_iq3, true)
BOUNDED_DOWN(bounded_down_iq2, false)
BOUNDED_DOWN(bounded_down_iq3, true)

