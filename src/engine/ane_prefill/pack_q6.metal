// Q6_K decoding follows the product dense tile loader, attributed to
// mlx-kquant (MIT) and ggml in the product THIRD-PARTY-NOTICES.
#include <metal_stdlib>
using namespace metal;

inline ushort q6_fp16(ushort bits,device atomic_uint* status) {
    uint sign=uint(bits)&0x8000u, exponent=(uint(bits)>>7)&255u, fraction=uint(bits)&127u;
    if(exponent==255u) { atomic_fetch_or_explicit(status,1u,memory_order_relaxed);return 0; }
    if(exponent>=143u) { atomic_fetch_or_explicit(status,2u,memory_order_relaxed);return 0; }
    if(exponent>=113u) return ushort(sign|((exponent-112u)<<10)|(fraction<<3));
    if(exponent<102u) return ushort(sign);
    uint mantissa=128u+fraction;
    if(exponent>=110u) return ushort(sign|(mantissa<<(exponent-110u)));
    uint shift=110u-exponent,rounded=mantissa>>shift,remainder=mantissa&((1u<<shift)-1u),halfway=1u<<(shift-1u);
    rounded+=remainder>halfway || (remainder==halfway && (rounded&1u));
    return ushort(sign|rounded);
}

// Integer expansion retains finite half subnormals before FP32 products.
inline float q6_half_float(ushort bits,device atomic_uint* status) {
    uint sign=(uint(bits)&0x8000u)<<16,exponent=(uint(bits)>>10)&31u,fraction=uint(bits)&1023u;
    if(exponent==31u) { atomic_fetch_or_explicit(status,8u,memory_order_relaxed);return 0.0f; }
    uint word;
    if(exponent!=0u) word=sign|((exponent+112u)<<23)|(fraction<<13);
    else if(fraction==0u) word=sign;
    else { uint leading=31u-clz(fraction);word=sign|((103u+leading)<<23)|((fraction-(1u<<leading))<<(23u-leading)); }
    return as_type<float>(word);
}

inline ushort q6_weight(device const uchar* weights,uint row,uint k,uint K,device atomic_uint* status) {
    device const uchar* block=weights+ulong(row)*(K/256)*210+(k/256)*210;
    uint sub=k%256,half_index=sub/128,quarter=(sub%128)/32,lane=sub%32;
    uchar low=block[half_index*64+(quarter%2)*32+lane],high=block[128+half_index*32+lane];
    int code=int(((low>>(quarter>=2?4:0))&15)|(((high>>(quarter*2))&3)<<4))-32;
    int scale=int(reinterpret_cast<device const char*>(block+192)[sub/16]);
    ushort d=ushort(block[208])|(ushort(block[209])<<8);
    float scaled=q6_half_float(d,status)*float(scale);
    float value=scaled*float(code);
    uint bits=as_type<uint>(value);
    if((bits&0x7f800000u)==0x7f800000u) { atomic_fetch_or_explicit(status,16u,memory_order_relaxed);return 0; }
    ushort bf16=ushort((bits+0x7fffu+((bits>>16)&1u))>>16);
    return q6_fp16(bf16,status);
}

template<bool QsaGate,bool Shared=false>
inline void q6_combined_body(device const ushort* activation,
    device const uchar* weights,device ushort* combined,
    device atomic_uint* status,constant uint* dims,uint group,uint tid,
    threadgroup ushort* tile,device const uchar* up) {
    uint rows=dims[0],channels=dims[1],outputs=dims[2],useful=dims[3],column_tiles=(channels+31u)/32u;
    uint activation_tiles=((rows+31u)/32u)*column_tiles;
    bool weight=group>=activation_tiles;
    uint local_group=weight?group-activation_tiles:group;
    uint row_base=(local_group/column_tiles)*32u,channel_base=(local_group%column_tiles)*32u;
    uint limit=weight?outputs:rows,x=tid%32u,y=tid/32u;
    for(uint offset=0;offset<32u;offset+=8u) {
        uint row=row_base+y+offset,channel=channel_base+x;
        ushort value=0;
        if(row<limit && channel<channels) {
            if(!weight) value=row<useful?q6_fp16(activation[row*channels+channel],status):ushort(0);
            else if(!Shared || row<1280u) value=q6_weight(Shared && row>=640u?up:weights,
                Shared?row%640u:(QsaGate ? (row/256u)*512u+256u+row%256u : row),channel,channels,status);
        }
        tile[(y+offset)*33u+x]=value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint offset=0;offset<32u;offset+=8u) {
        uint row=row_base+x,channel=channel_base+y+offset;
        if(row<limit && channel<channels) combined[channel*(rows+outputs)+(weight?rows:0u)+row]=tile[x*33u+y+offset];
    }
}

#define Q6_PACK(NAME,GATE) \
kernel void NAME(device const ushort* a [[buffer(0)]],device const uchar* w [[buffer(1)]], \
    device ushort* c [[buffer(2)]],device atomic_uint* s [[buffer(3)]],constant uint* d [[buffer(4)]], \
    uint g [[threadgroup_position_in_grid]],uint t [[thread_position_in_threadgroup]]) { \
    threadgroup ushort tile[32*33];q6_combined_body<GATE>(a,w,c,s,d,g,t,tile,w); \
}
Q6_PACK(q6_combined_pack,false)
Q6_PACK(q6_qsa_gate_pack,true)

kernel void q6_shared_front_pack(device const ushort* a [[buffer(0)]],device const uchar* w [[buffer(1)]],
    device ushort* c [[buffer(2)]],device atomic_uint* s [[buffer(3)]],device const uchar* up [[buffer(4)]],
    constant uint* d [[buffer(5)]],uint g [[threadgroup_position_in_grid]],uint t [[thread_position_in_threadgroup]]) {
    threadgroup ushort tile[32*33];q6_combined_body<false,true>(a,w,c,s,d,g,t,tile,up);
}
