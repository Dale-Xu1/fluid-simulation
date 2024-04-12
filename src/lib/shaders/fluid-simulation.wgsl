const DT: f32 = 1;
const SCALE: f32 = 3;

const VELOCITY_DIFFUSION: f32 = 0.998;
const DENSITY_DIFFUSION: f32 = 0.99;

const CURL: f32 = 0.12;

@group(0) @binding(0) var texture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<uniform> dimensions: vec2u;
@group(0) @binding(2) var<uniform> densityDimensions: vec2u;

@group(1) @binding(0) var<storage, read_write> velocity: array<vec2f>;
@group(1) @binding(1) var<storage> previousVelocity: array<vec2f>;

@group(1) @binding(2) var<storage, read_write> divergence: array<f32>;
@group(1) @binding(3) var<storage, read_write> curl: array<f32>;

@group(1) @binding(4) var<storage, read_write> pressure: array<f32>;
@group(1) @binding(5) var<storage> previousPressure: array<f32>;

@group(1) @binding(6) var<storage, read_write> density: array<vec3f>;
@group(1) @binding(7) var<storage> previousDensity: array<vec3f>;

// Helper function to index buffer using two dimensional index
fn indexu(id: vec2u) -> u32 { return id.y * dimensions.x + id.x; }
fn index(id: vec2i) -> u32
{
    let d = vec2i(dimensions);
    return indexu(vec2u((id + d) % d));
}

// Bilinear interpolation functions
fn bilinear2f(data: ptr<storage, array<vec2f>>, id: vec2f) -> vec2f
{
    let i = vec2i(floor(id)); let ij = vec4i(i, i + 1);
    let t = id - vec2f(i);

    return mix(
        mix(data[index(ij.xy)], data[index(ij.xw)], t.y),
        mix(data[index(ij.zy)], data[index(ij.zw)], t.y), t.x);
}

@compute @workgroup_size(8, 8)
fn advection(@builtin(global_invocation_id) id: vec3u)
{
    if (id.x >= dimensions.x || id.y >= dimensions.y) { return; }

    // Move velocity vectors based on velocity field
    let i = indexu(id.xy);
    let previous = vec2f(id.xy) - previousVelocity[i] * DT;

    // Interpolation of four pixels surrounding previous position
    velocity[i] = VELOCITY_DIFFUSION * bilinear2f(&previousVelocity, previous);
}

@compute @workgroup_size(8, 8)
fn calculateDivergence(@builtin(global_invocation_id) id: vec3u)
{
    if (id.x >= dimensions.x || id.y >= dimensions.y) { return; }

    let i = vec2i(id.xy);
    let u = velocity[index(i + vec2i(0, 1))].y;
    let d = velocity[index(i - vec2i(0, 1))].y;
    let r = velocity[index(i + vec2i(1, 0))].x;
    let l = velocity[index(i - vec2i(1, 0))].x;

    divergence[indexu(id.xy)] = 0.5 * (u - d + r - l);
}

@compute @workgroup_size(8, 8)
fn calculateCurl(@builtin(global_invocation_id) id: vec3u)
{
    if (id.x >= dimensions.x || id.y >= dimensions.y) { return; }

    let i = vec2i(id.xy);
    let u = velocity[index(i + vec2i(0, 1))].x;
    let d = velocity[index(i - vec2i(0, 1))].x;
    let r = velocity[index(i + vec2i(1, 0))].y;
    let l = velocity[index(i - vec2i(1, 0))].y;

    curl[indexu(id.xy)] = d - u + l - r;
}

@compute @workgroup_size(8, 8)
fn vorticity(@builtin(global_invocation_id) id: vec3u)
{
    if (id.x >= dimensions.x || id.y >= dimensions.y) { return; }

    let position = vec2i(id.xy);
    let i = indexu(id.xy);

    let m = curl[i];
    let u = curl[index(position + vec2i(0, 1))];
    let d = curl[index(position - vec2i(0, 1))];
    let r = curl[index(position + vec2i(1, 0))];
    let l = curl[index(position - vec2i(1, 0))];

    let force = vec2f(abs(u) - abs(d), abs(r) - abs(l));
    let ls = max(0.001, dot(force, force));

    velocity[i] += CURL * m * force * inverseSqrt(ls) * DT;
}

@compute @workgroup_size(8, 8)
fn jacobi(@builtin(global_invocation_id) id: vec3u)
{
    if (id.x >= dimensions.x || id.y >= dimensions.y) { return; }

    let position = vec2i(id.xy);
    let i = indexu(id.xy);

    let u = previousPressure[index(position + vec2i(0, 1))];
    let d = previousPressure[index(position - vec2i(0, 1))];
    let r = previousPressure[index(position + vec2i(1, 0))];
    let l = previousPressure[index(position - vec2i(1, 0))];
    let div = divergence[i];

    pressure[i] = 0.25 * (u + d + r + l - div);
}

@compute @workgroup_size(8, 8)
fn gradientSubtraction(@builtin(global_invocation_id) id: vec3u)
{
    if (id.x >= dimensions.x || id.y >= dimensions.y) { return; }

    let position = vec2i(id.xy);
    let u = pressure[index(position + vec2i(0, 1))];
    let d = pressure[index(position - vec2i(0, 1))];
    let r = pressure[index(position + vec2i(1, 0))];
    let l = pressure[index(position - vec2i(1, 0))];

    velocity[indexu(id.xy)] -= 0.5 * vec2f(r - l, u - d) * DT;
}

// Helper function to index buffer using two dimensional index
fn sindexu(id: vec2u) -> u32 { return id.y * densityDimensions.x + id.x; }
fn sindex(id: vec2i) -> u32
{
    let d = vec2i(densityDimensions);
    return sindexu(vec2u((id + d) % d));
}

fn bilinear3f(data: ptr<storage, array<vec3f>>, id: vec2f) -> vec3f
{
    let i = vec2i(floor(id)); let ij = vec4i(i, i + 1);
    let t = id - vec2f(i);

    return mix(
        mix(data[sindex(ij.xy)], data[sindex(ij.xw)], t.y),
        mix(data[sindex(ij.zy)], data[sindex(ij.zw)], t.y), t.x);
}

@compute @workgroup_size(8, 8)
fn densityAdvection(@builtin(global_invocation_id) id: vec3u)
{
    if (id.x >= densityDimensions.x || id.y >= densityDimensions.y) { return; }

    // Move density based on velocity field
    let i = vec2f(id.xy);
    let previous = vec2f(id.xy) - SCALE * bilinear2f(&previousVelocity, i / SCALE) * DT;

    density[sindexu(id.xy)] = DENSITY_DIFFUSION * bilinear3f(&previousDensity, previous);
}

@compute @workgroup_size(8, 8)
fn transfer(@builtin(global_invocation_id) id: vec3u)
{
    // Transfer density data as colors to texture
    let color = vec4f(density[sindexu(id.xy)], 1);
    textureStore(texture, id.xy, color);
}

// User input passes
@group(2) @binding(0) var<uniform> position: vec2f;
@group(2) @binding(1) var<uniform> impulse: vec2f;
@group(2) @binding(2) var<uniform> color: vec3f;
@group(2) @binding(3) var<uniform> radius: f32;

@compute @workgroup_size(8, 8)
fn addImpulse(@builtin(global_invocation_id) id: vec3u)
{
    let offset = vec2f(id.xy) - vec2f(radius);
    if (dot(offset, offset) > radius * radius)  { return; }

    let i = index(vec2i(position + offset));
    velocity[i] += impulse;
}

@compute @workgroup_size(8, 8)
fn addDensity(@builtin(global_invocation_id) id: vec3u)
{
    let offset = vec2f(id.xy) - vec2f(radius);
    if (dot(offset, offset) > radius * radius)  { return; }

    let i = sindex(vec2i(SCALE * position + offset));
    density[i] += color;
}
