const DT: f32 = 1;

const DIFF: f32 = 0.99;
const CURL: f32 = 0.8;

const VELOCITY: f32 = 0.5;
const DENSITY: f32 = 0.2;

@group(0) @binding(0) var texture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<uniform> dimensions: vec2u;

@group(0) @binding(2) var<storage, read_write> velocity: array<vec2f>;
@group(0) @binding(3) var<storage> previousVelocity: array<vec2f>;

@group(0) @binding(4) var<storage, read_write> divergence: array<f32>;
@group(0) @binding(5) var<storage, read_write> pressure: array<f32>;
@group(0) @binding(6) var<storage> previousPressure: array<f32>;

@group(0) @binding(7) var<storage, read_write> density: array<vec4f>;
@group(0) @binding(8) var<storage> previousDensity: array<vec4f>;

// Helper function to index buffer using two dimensional index
fn index(id: vec2u) -> u32 { return (id.y % dimensions.y) * dimensions.x + (id.x % dimensions.x); }

// Bilinear interpolation functions
fn bilinear2f(data: ptr<storage, array<vec2f>>, id: vec2f) -> vec2f
{
    let i = vec2u(id);
    let ij = vec4u(i, i + 1);
    let t = id - vec2f(ij.xy);

    return mix(
        mix(data[index(ij.xy)], data[index(ij.xw)], t.y),
        mix(data[index(ij.zy)], data[index(ij.zw)], t.y), t.x);
}

fn bilinear4f(data: ptr<storage, array<vec4f>>, id: vec2f) -> vec4f
{
    let i = vec2u(id);
    let ij = vec4u(i, i + 1);
    let t = id - vec2f(ij.xy);

    return mix(
        mix(data[index(ij.xy)], data[index(ij.xw)], t.y),
        mix(data[index(ij.zy)], data[index(ij.zw)], t.y), t.x);
}

@compute @workgroup_size(8, 8)
fn advection(@builtin(global_invocation_id) id: vec3u)
{
    // Interpolation of four pixels surrounding previous position
    let i = index(id.xy);
    let previous = vec2f(id.xy) - previousVelocity[i] * DT;

    velocity[i] = DIFF * bilinear2f(&previousVelocity, previous);
}

@compute @workgroup_size(8, 8)
fn transfer(@builtin(global_invocation_id) id: vec3u)
{
    textureStore(texture, id.xy, density[index(id.xy)]);
}
