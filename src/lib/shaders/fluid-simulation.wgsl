const DT: f32 = 1;

const DIFFUSION: f32 = 0.99;
const CURL: f32 = 0.8;

@group(0) @binding(0) var texture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<uniform> dimensions: vec2u;

@group(0) @binding(2) var<storage, read_write> velocity: array<vec2f>;
@group(0) @binding(3) var<storage> previousVelocity: array<vec2f>;

@group(0) @binding(4) var<storage, read_write> divergence: array<f32>;
@group(0) @binding(5) var<storage, read_write> pressure: array<f32>;
@group(0) @binding(6) var<storage> previousPressure: array<f32>;

@group(0) @binding(7) var<storage, read_write> density: array<vec3f>;
@group(0) @binding(8) var<storage> previousDensity: array<vec3f>;

// Helper function to index buffer using two dimensional index
fn index(id: vec2u) -> u32 { return (id.y % dimensions.y) * dimensions.x + (id.x % dimensions.x); }
fn indexi(id: vec2i) -> u32 { return index(vec2u(id + vec2i(dimensions))); }

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

fn bilinear3f(data: ptr<storage, array<vec3f>>, id: vec2f) -> vec3f
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
    // Move velocity vectors based on velocity field
    let i = index(id.xy);
    let previous = vec2f(id.xy) - previousVelocity[i] * DT;

    // Interpolation of four pixels surrounding previous position
    velocity[i] = DIFFUSION * bilinear2f(&previousVelocity, previous);
}

@compute @workgroup_size(8, 8)
fn densityAdvection(@builtin(global_invocation_id) id: vec3u)
{
    // Move density based on velocity field
    let i = index(id.xy);
    let previous = vec2f(id.xy) - previousVelocity[i] * DT;

    density[i] = DIFFUSION * bilinear3f(&previousDensity, previous);
}

@compute @workgroup_size(8, 8)
fn transfer(@builtin(global_invocation_id) id: vec3u)
{
    // Transfer density data as colors to texture
    let color = vec4f(density[index(id.xy)], 1);
    textureStore(texture, id.xy, color);
}

// User input passes
const RADIUS: f32 = 3;

@group(1) @binding(0) var<uniform> position: vec2f;
@group(1) @binding(1) var<uniform> impulse: vec2f;
@group(1) @binding(2) var<uniform> color: vec3f;

@compute @workgroup_size(8, 8)
fn addImpulse(@builtin(global_invocation_id) id: vec3u)
{
    let offset = vec2f(id.xy) - vec2f(RADIUS);
    if (dot(offset, offset) > RADIUS * RADIUS)  { return; }

    velocity[indexi(vec2i(position + offset))] += impulse;
}

@compute @workgroup_size(8, 8)
fn addDensity(@builtin(global_invocation_id) id: vec3u)
{
    let offset = vec2f(id.xy) - vec2f(RADIUS);
    if (dot(offset, offset) > RADIUS * RADIUS)  { return; }

    density[indexi(vec2i(position + offset))] += color;
}
