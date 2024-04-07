struct VertexInput
{
    @location(0) position: vec2f,
    @builtin(instance_index) instance: u32
};

struct FragmentInput
{
    @builtin(position) position: vec4f
};

@group(0) @binding(0) var<uniform> grid: vec2f;
@group(0) @binding(1) var<storage> stateBuffer: array<u32>;

@vertex
fn vs(input: VertexInput) -> FragmentInput
{
    let i = f32(input.instance);
    let cell = vec2f(i % grid.x, floor(i / grid.x));
    let state = f32(stateBuffer[input.instance]);

    let offset = cell / grid * 2;
    let position = (input.position * state + 1) / grid - 1 + offset;

    var output: FragmentInput;
    output.position = vec4f(position, 0, 1);

    return output;
}

@fragment
fn fs() -> @location(0) vec4f
{
    return vec4f(1);
}
