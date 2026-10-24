#include <metal_stdlib>
using namespace metal;

// Integer conversion retains finite half subnormals before BF16 rounding.
inline ushort tail_bf16(ushort bits, device atomic_uint* status) {
    uint sign=(uint(bits)&0x8000u)<<16,exponent=(uint(bits)>>10)&31u,fraction=uint(bits)&1023u;
    if(exponent==31u) { atomic_fetch_or_explicit(status,4u,memory_order_relaxed);return 0; }
    uint word;
    if(exponent!=0u) word=sign|((exponent+112u)<<23)|(fraction<<13);
    else if(fraction==0u) word=sign;
    else { uint leading=31u-clz(fraction);word=sign|((103u+leading)<<23)|((fraction-(1u<<leading))<<(23u-leading)); }
    return ushort((word+0x7fffu+((word>>16)&1u))>>16);
}

template<bool Raw>
inline void tail_unpack(device const ushort* source,device ushort* destination,
    device atomic_uint* status,constant uint* dims,uint group,uint tid,threadgroup ushort* tile) {
    uint physical=dims[0],rows=dims[1],channels=dims[2],column_tiles=(channels+31u)/32u;
    uint row_base=(group/column_tiles)*32u,channel_base=(group%column_tiles)*32u,x=tid%32u,y=tid/32u;
    for(uint offset=0;offset<32u;offset+=8u) {
        uint row=row_base+x,channel=channel_base+y+offset;
        tile[(y+offset)*33u+x]=row<rows && channel<channels?source[channel*physical+row]:ushort(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint offset=0;offset<32u;offset+=8u) {
        uint row=row_base+y+offset,channel=channel_base+x;
        if(row<rows && channel<channels) {
            ushort value=tile[x*33u+y+offset];
            destination[row*channels+channel]=Raw?value:tail_bf16(value,status);
        }
    }
}
#define TAIL_UNPACK(NAME,RAW) \
kernel void NAME(device const ushort* source [[buffer(0)]],device ushort* destination [[buffer(1)]], \
    device atomic_uint* status [[buffer(2)]],constant uint* dims [[buffer(3)]], \
    uint group [[threadgroup_position_in_grid]],uint tid [[thread_position_in_threadgroup]]) { \
    threadgroup ushort tile[32*33];tail_unpack<RAW>(source,destination,status,dims,group,tid,tile); \
}
TAIL_UNPACK(tail_unpack_bf16,false)
TAIL_UNPACK(tail_unpack_raw,true)

kernel void tail_unpack_shared(device const ushort* source [[buffer(0)]],device ushort* gate [[buffer(1)]],
    device ushort* up [[buffer(2)]],device atomic_uint* status [[buffer(3)]],constant uint* dims [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],uint tid [[thread_position_in_threadgroup]]) {
    uint physical=dims[0],rows=dims[1],channels=dims[2],width=channels/2u,column_tiles=(channels+31u)/32u;
    uint row_base=(group/column_tiles)*32u,channel_base=(group%column_tiles)*32u,x=tid%32u,y=tid/32u;
    threadgroup ushort tile[32*33];
    for(uint offset=0;offset<32u;offset+=8u) {
        uint row=row_base+x,channel=channel_base+y+offset;
        tile[(y+offset)*33u+x]=row<rows && channel<channels?source[channel*physical+row]:ushort(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint offset=0;offset<32u;offset+=8u) {
        uint row=row_base+y+offset,channel=channel_base+x;
        if(row<rows && channel<channels) (channel<width?gate:up)[row*width+channel%width]=tail_bf16(tile[x*33u+y+offset],status);
    }
}
