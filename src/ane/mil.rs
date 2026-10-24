//! Dynamic matmul graph adapted from maderix/ANE under the MIT license.

use super::{DynamicLayout, ProjectionShape, Result};

pub(crate) fn mil(shape: ProjectionShape) -> Result<String> {
    let layout = DynamicLayout::new(shape)?;
    let ProjectionShape {
        rows,
        input_channels: ic,
        output_channels: oc,
    } = shape;
    let spatial = layout.spatial;
    Ok(format!(
        r#"program(1.3)
[buildInfo = dict<string, string>({{{{"coremlc-component-MIL", "3510.2.1"}}, {{"coremlc-version", "3505.4.1"}}, {{"coremltools-component-milinternal", ""}}, {{"coremltools-version", "9.0"}}}})]
{{
    func main<ios18>(tensor<fp16, [1, {ic}, 1, {spatial}]> x) {{
        tensor<int32, [4]> ba = const()[name=string("ba"), val=tensor<int32, [4]>([0,0,0,0])];
        tensor<int32, [4]> sa = const()[name=string("sa"), val=tensor<int32, [4]>([1,{ic},1,{rows}])];
        tensor<fp16, [1,{ic},1,{rows}]> act = slice_by_size(x=x,begin=ba,size=sa)[name=string("act")];
        tensor<int32, [4]> bw = const()[name=string("bw"), val=tensor<int32, [4]>([0,0,0,{rows}])];
        tensor<int32, [4]> sw = const()[name=string("sw"), val=tensor<int32, [4]>([1,{ic},1,{oc}])];
        tensor<fp16, [1,{ic},1,{oc}]> wt = slice_by_size(x=x,begin=bw,size=sw)[name=string("wt")];
        tensor<int32, [4]> ra = const()[name=string("ra"), val=tensor<int32, [4]>([1,1,{ic},{rows}])];
        tensor<fp16, [1,1,{ic},{rows}]> a2 = reshape(shape=ra,x=act)[name=string("a2")];
        tensor<int32, [4]> pm = const()[name=string("pm"), val=tensor<int32, [4]>([0,1,3,2])];
        tensor<fp16, [1,1,{rows},{ic}]> a3 = transpose(perm=pm,x=a2)[name=string("a3")];
        tensor<int32, [4]> rw = const()[name=string("rw"), val=tensor<int32, [4]>([1,1,{ic},{oc}])];
        tensor<fp16, [1,1,{ic},{oc}]> W = reshape(shape=rw,x=wt)[name=string("W")];
        bool bF = const()[name=string("bF"), val=bool(false)];
        tensor<fp16, [1,1,{rows},{oc}]> yh = matmul(transpose_x=bF,transpose_y=bF,x=a3,y=W)[name=string("mm")];
        tensor<fp16, [1,1,{oc},{rows}]> yt = transpose(perm=pm,x=yh)[name=string("yt")];
        tensor<int32, [4]> ro = const()[name=string("ro"), val=tensor<int32, [4]>([1,{oc},1,{rows}])];
        tensor<fp16, [1,{oc},1,{rows}]> y = reshape(shape=ro,x=yt)[name=string("y")];
    }} -> (y);
}}
"#
    ))
}
