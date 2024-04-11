struct VertexInput
{
    @location(0) position: vec2f
};

struct FragmentInput
{
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f
};

@group(0) @binding(0) var textureSampler: sampler;
@group(0) @binding(1) var texture: texture_2d<f32>;

@vertex
fn vs(input: VertexInput) -> FragmentInput
{
    var output: FragmentInput;
    output.position = vec4f(input.position, 0, 1);
    output.uv = 0.5 * input.position + 0.5;

    return output;
}

@fragment
fn fs(input: FragmentInput) -> @location(0) vec4f
{
    return textureSample(texture, textureSampler, input.uv);
}
